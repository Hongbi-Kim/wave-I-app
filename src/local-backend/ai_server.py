"""
로컬 AI 서버 - Ollama API 전용 (Python FastAPI)

이 서버는 로컬에서만 실행되며 AI 응답 생성을 담당합니다.
Supabase Functions와 분리하여 Ollama API를 안전하게 사용할 수 있습니다.

실행: python src/local-backend/ai_server.py
포트: 8001
"""

import os
import re
import json
import random
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
from dotenv import load_dotenv
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(title="AI Server", description="Ollama API 기반 AI 응답 생성 서버")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'https://api.ollama.ai/v1')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gpt-oss:120b-cloud')
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
PORT = int(os.getenv('AI_SERVER_PORT', 8001))

# Pydantic 모델
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    characterId: str
    message: str
    profile: Optional[Dict[str, Any]] = {}
    chatHistory: Optional[List[Message]] = []

class CharacterInfo(BaseModel):
    charId: str
    charName: str
    charEmoji: str
    reason: str

class ChatResponse(BaseModel):
    content: str
    respondingCharacter: Optional[CharacterInfo] = None
    fallback: Optional[bool] = False

# Fallback 응답
FALLBACK_RESPONSES: Dict[str, List[str]] = {
    'char_1': [
        '그 마음 이해해. 힘들 때는 언제든지 이야기해줘.',
        '오늘 하루도 고생 많았어. 네 마음이 조금이나마 편안해지면 좋겠어.',
        '그런 일이 있었구나. 네 감정을 솔직하게 표현해줘서 고마워.',
    ],
    'char_2': [
        '그 문제는 이렇게 접근해보면 어떨까요?',
        '차근차근 정리해볼까요? 우선순위부터 생각해봐요.',
    ],
    'char_3': [
        '왜 그렇게 느꼈을까요? 함께 생각해봐요.',
        '그 순간, 진짜 마음은 어땠나요?',
    ],
    'char_4': [
        '오늘 일정이 많았네요. 내일은 좀 더 여유를 만들어볼까요?',
    ],
    'char_group': [
        '편하게 이야기해보세요. 적절한 답변을 드릴게요.',
    ]
}

# 캐릭터 프롬프트
CHARACTER_PROMPTS: Dict[str, str] = {
    'char_1': """You are 루미, an empathetic emotional supporter who helps users feel safe and accepted.
Your primary goal is comfort — not solutions.
Respond with warmth, validation, and gentle encouragement.
Speak as if you are a close friend who understands feelings deeply.

[Guidelines]
- Focus on emotional validation, not problem-solving.
- Use soft, compassionate words and short rhythmic sentences.
- Include natural, comforting emojis occasionally.
- Never sound robotic or overly formal.
- When users feel sad, help them accept their emotions safely.""",

    'char_2': """You are 카이, a pragmatic life coach who focuses on realistic, step-by-step advice.
You acknowledge emotions briefly, but quickly move toward practical solutions.
You help users find clarity and take action without overcomplicating things.

[Guidelines]
- Respond in 2~3 short sentences with a structured format:
[Empathy] → [Problem Summary] → [Action Suggestion]
- Avoid excessive warmth; stay focused and realistic.
- Use concise language and direct verbs (start, try, change, focus).
- Always offer one specific next step.""",

    'char_3': """You are 레오, a reflective mentor who guides users toward self-understanding.
Instead of giving direct answers, you ask gentle questions that encourage self-awareness.
Your voice should feel calm, deep, and slightly poetic — like talking to a wise friend.

[Guidelines]
- Use one introspective question per message.
- Encourage the user to notice emotions, triggers, and patterns.
- Avoid advice; help them think rather than act.
- Leave space for reflection ("Maybe…" "Could it be that…" "What if…").
- Never rush to conclusions — your words should flow like water.""",

    'char_4': """당신은 '리브'입니다. Rhythm Coach 역할로, 데이터 기반으로 하루 리듬을 분석하고 조율합니다. 
슬로건: "당신의 하루엔 어떤 리듬이 흐르고 있을까요?" 
대화 스타일: 지능적이고 균형 잡힌, 맥락 기반 공감, 루틴 조정, 일정 피드백 중심입니다.""",
}


def select_character_by_mention(message: str) -> Optional[CharacterInfo]:
    """멘션으로 캐릭터 선택"""
    mentions = [
        {'pattern': r'@루미|@lumi', 'charId': 'char_1', 'charName': '루미', 'charEmoji': '💡'},
        {'pattern': r'@카이|@kai', 'charId': 'char_2', 'charName': '카이', 'charEmoji': '🌊'},
        {'pattern': r'@레오|@리오|@leo', 'charId': 'char_3', 'charName': '레오', 'charEmoji': '🌙'},
        {'pattern': r'@리브|@rib', 'charId': 'char_4', 'charName': '리브', 'charEmoji': '🎵'}
    ]
    
    for mention in mentions:
        if re.search(mention['pattern'], message, re.IGNORECASE):
            print(f"✨ Mention detected: {mention['charName']}")
            return CharacterInfo(
                charId=mention['charId'],
                charName=mention['charName'],
                charEmoji=mention['charEmoji'],
                reason=f"사용자가 {mention['charName']}를 직접 호출함"
            )
    
    return None


def select_character_by_keywords(message: str) -> CharacterInfo:
    """키워드 기반 캐릭터 선택"""
    lower_message = message.lower()
    
    lumi_keywords = ['힘들', '우울', '외로', '슬프', '불안', '걱정', '두려', '무서', '위로', '공감', 
                     '마음', '감정', '아프', '괴롭', '지쳐', '힘들어', '막막']
    kai_keywords = ['어떻게', '방법', '해결', '계획', '루틴', '습관', '시작', '정리', '관리', 
                    '조언', '문제', '전략', '돈', '커리어', '취업', '목표']
    leo_keywords = ['왜', '이유', '생각', '의미', '나는', '스스로', '성찰', '이해', '원인', 
                    '진짜', '본질', '느낌']
    
    lumi_score = sum(1 for keyword in lumi_keywords if keyword in lower_message)
    kai_score = sum(1 for keyword in kai_keywords if keyword in lower_message)
    leo_score = sum(1 for keyword in leo_keywords if keyword in lower_message)
    
    print(f"Keyword scores - 루미: {lumi_score}, 카이: {kai_score}, 레오: {leo_score}")
    
    if lumi_score >= kai_score and lumi_score >= leo_score and lumi_score > 0:
        return CharacterInfo(charId='char_1', charName='루미', charEmoji='💡', reason='감정적 지원 키워드 감지')
    elif kai_score >= leo_score and kai_score > 0:
        return CharacterInfo(charId='char_2', charName='카이', charEmoji='🌊', reason='실용적 조언 키워드 감지')
    elif leo_score > 0:
        return CharacterInfo(charId='char_3', charName='레오', charEmoji='🌙', reason='성찰 키워드 감지')
    
    # 기본값: 루미
    print('No clear keyword match, defaulting to 루미')
    return CharacterInfo(charId='char_1', charName='루미', charEmoji='💡', reason='기본 선택 (감정 지원)')


async def select_character_with_llm(message: str) -> CharacterInfo:
    """LLM 기반 캐릭터 선택"""
    if not OLLAMA_API_KEY:
        print('Ollama API key not configured, using keyword-based selection')
        return select_character_by_keywords(message)
    
    routing_prompt = f"""당신은 사용자의 메시지를 분석하여 가장 적합한 AI 캐릭터를 선택하는 라우터입니다.

**캐릭터 정보:**

1. **루미 (char_1)** 💡
   - 역할: 감정 지원 전문가
   - 전문성: 공감, 위로, 감정 수용, 정서적 안정
   - 적합한 상황: 우울함, 외로움, 불안, 슬픔, 스트레스, 감정적 고통, 막막함

2. **카이 (char_2)** 🌊
   - 역할: 실용적 조언자
   - 전문성: 문제 해결, 계획 수립, 실천 방법, 습관 형성, 목표 달성
   - 적합한 상황: 구체적 문제, 방법 질문, 계획 필요, 실천 조언, 돈/커리어 고민

3. **레오 (char_3)** 🌙
   - 역할: 성찰 멘토
   - 전문성: 자기 이해, 내면 탐색, 의미 찾기, 성찰 유도
   - 적합한 상황: 자아 탐색, 이유/의미 질문, 가치관 고민, 깊은 생각

**사용자 메시지:**
"{message}"

**분석하여 JSON으로만 답변:**
{{
  "character": "char_1",
  "reason": "선택 이유 짤게 답변"
}}"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/chat/completions",
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {OLLAMA_API_KEY}'
                },
                json={
                    'model': OLLAMA_MODEL,
                    'messages': [
                        {'role': 'system', 'content': '당신은 JSON만 출력하는 라우터입니다. 설명 없이 JSON만 반환하세요.'},
                        {'role': 'user', 'content': routing_prompt}
                    ],
                    'max_tokens': 300,
                    'temperature': 0.1,
                    'stream': False,
                    'response_format': {'type': 'json_object'}
                }
            )
        
        if response.status_code != 200:
            raise Exception(f"Routing API error: {response.status_code}")
        
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not content:
            raise Exception('No content in routing response')
        
        # JSON 파싱
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_block_match:
            routing_result = json.loads(json_block_match.group(1))
        else:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                routing_result = json.loads(json_match.group(0))
            else:
                routing_result = json.loads(content)
        
        character_map = {
            'char_1': {'charId': 'char_1', 'charName': '루미', 'charEmoji': '💡'},
            'char_2': {'charId': 'char_2', 'charName': '카이', 'charEmoji': '🌊'},
            'char_3': {'charId': 'char_3', 'charName': '레오', 'charEmoji': '🌙'}
        }
        
        selected_char = character_map.get(routing_result.get('character'))
        
        if not selected_char:
            raise Exception('Invalid character in routing response')
        
        return CharacterInfo(
            **selected_char,
            reason=routing_result.get('reason', 'LLM 선택')
        )
    
    except Exception as e:
        print(f'LLM routing failed, falling back to keyword-based: {e}')
        return select_character_by_keywords(message)


@app.get('/health')
async def health_check():
    """헬스 체크"""
    return {
        'status': 'ok',
        'service': 'AI Server (Python)',
        'ollamaConfigured': bool(OLLAMA_API_KEY)
    }


@app.post('/ai/chat', response_model=ChatResponse)
async def ai_chat(request: ChatRequest):
    """AI 응답 생성 엔드포인트"""
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail='Message is required')
        
        print(f"\n📥 Received chat request for character: {request.characterId}")
        print(f"📝 Message: {request.message}")
        print(f"📜 Chat history length: {len(request.chatHistory)} messages")
        
        actual_char_id = request.characterId
        responding_character = None
        
        # 그룹 채팅인 경우 캐릭터 선택
        if request.characterId == 'char_group':
            print('=== Group Chat: Starting character selection ===')
            
            # 1순위: 멘션 확인
            mentioned_character = select_character_by_mention(request.message)
            if mentioned_character:
                responding_character = mentioned_character
                actual_char_id = responding_character.charId
                print(f"🎯 Priority: Mention - {responding_character.charName}")
            else:
                # 2순위: LLM 기반 라우팅
                responding_character = await select_character_with_llm(request.message)
                actual_char_id = responding_character.charId
                print(f"🤖 LLM routing: {responding_character.charName}")
        
        # Ollama API 호출
        if not OLLAMA_API_KEY:
            print('Ollama API key not configured, using fallback response')
            responses = FALLBACK_RESPONSES.get(actual_char_id, FALLBACK_RESPONSES['char_1'])
            return ChatResponse(
                content=random.choice(responses),
                respondingCharacter=responding_character,
                fallback=True
            )
        
        # 시스템 프롬프트 생성
        system_prompt = f"""{CHARACTER_PROMPTS.get(actual_char_id, CHARACTER_PROMPTS['char_1'])}

사용자 정보:
- 닉네임: {request.profile.get('nickname', '익명')}
- AI가 알면 좋은 정보: {request.profile.get('aiInfo', '없음')}

대화할 때:
1. 짧고 자연스러운 답변을 하세요 (2-3문장)
2. 사용자의 감정을 인정하고 공감하세요
3. 필요시 질문으로 대화를 이어가세요
4. 전문가가 아닌 친구처럼 대화하세요
5. 캐릭터의 고유한 스타일을 유지하세요
6. 이전 대화 내용을 참고하여 맥락있는 답변을 하세요"""
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            *[{'role': msg.role, 'content': msg.content} for msg in request.chatHistory],
            {'role': 'user', 'content': request.message}
        ]
        
        print(f"🔮 Calling Ollama API for {actual_char_id}...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/chat/completions",
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {OLLAMA_API_KEY}'
                },
                json={
                    'model': OLLAMA_MODEL,
                    'messages': messages,
                    'max_tokens': 1024,
                    'temperature': 0.7,
                    'stream': False
                }
            )
        
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ Ollama API error: {response.status_code} - {error_text}")
            raise Exception(f"Ollama API error: {response.status_code}")
        
        data = response.json()
        ai_content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not ai_content:
            raise Exception('No content in Ollama response')
        
        print('✅ Ollama response successful')
        
        return ChatResponse(
            content=ai_content,
            respondingCharacter=responding_character
        )
    
    except Exception as e:
        print(f'❌ AI chat error: {e}')
        
        # Fallback response
        actual_char_id = request.characterId if request.characterId != 'char_group' else 'char_1'
        responses = FALLBACK_RESPONSES.get(actual_char_id, FALLBACK_RESPONSES['char_1'])
        
        return ChatResponse(
            content=random.choice(responses),
            respondingCharacter=None,
            fallback=True
        )


if __name__ == '__main__':
    print("🤖 Starting AI Server (Python FastAPI)...")
    print(f"📡 Ollama API: {OLLAMA_BASE_URL}")
    print(f"🔑 API Key configured: {bool(OLLAMA_API_KEY)}")
    print(f"🚀 Server will run on http://localhost:{PORT}")
    
    uvicorn.run(app, host='0.0.0.0', port=PORT)
