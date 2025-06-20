from dotenv import load_dotenv
import os

# 環境変数からAPIキーをロード
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- Streamlit Session State の初期化 ---
# セッションステートに 'trainer_memory' が存在しない場合、初期化する
if "trainer_memory" not in st.session_state:
    st.session_state.trainer_memory = ConversationBufferMemory(memory_key="chat_history")

# セッションステートに 'doctor_memory' が存在しない場合、初期化する
if "doctor_memory" not in st.session_state:
    st.session_state.doctor_memory = ConversationBufferMemory(memory_key="chat_history")

# セッションステートに 'llm' が存在しない場合、初期化する
if "llm" not in st.session_state:
    # OpenAI Chatモデルを初期化し、セッションステートに保存
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=api_key)

# --- LLMとの対話ロジック ---
def unified_prompt(input_message: str, selected_item: str) -> str:
    """
    selected_itemの内容によってプロンプトを切り替え、AI応答を生成する関数

    Args:
        input_message (str): ユーザーからの質問や相談内容
        selected_item (str): 選択された項目（'医師', 'スポーツトレーナー' ）

    Returns:
        str: AIによる応答メッセージ
    """
    # 選択された専門家に基づいて、使用するメモリを決定
    if selected_item == "医師":
        current_memory = st.session_state.doctor_memory
        template = """
        あなたは優秀な医師です。
        ユーザの体調や症状に関する質問に対して、病気の可能性や健康診断の結果について専門的な知識に基づき回答してください。
        会話履歴：
        {chat_history}
        入力：{input_message}
        出力：
        """
    elif selected_item == "スポーツトレーナー":
        current_memory = st.session_state.trainer_memory
        template = """
        あなたは優秀なスポーツトレーナーです。
        ユーザの質問に対して、専門的な知識に基づき、トレーニングメニューやそのトレーニングメニューに合った食事のアドバイスを提供してください。
        食事における各食材が体に与える影響についても言及してください。
        会話履歴：
        {chat_history}
        入力：{input_message}
        出力：
        """
    else:
        # 予期しないselected_itemの場合の処理 (エラーハンドリング)
        st.error("無効な専門家が選択されました。")
        return ""

    # プロンプトテンプレートを作成
    prompt = PromptTemplate(
        input_variables=["chat_history", "input_message"],
        template=template
    )

    # ConversationChainを初期化する際に、selected_itemに応じたメモリを使用
    chain = ConversationChain(
        llm=st.session_state.llm,
        prompt=prompt,
        memory=current_memory, # ここで適切なメモリを使用
        input_key="input_message", # プロンプトのinput_variablesに合わせておく
        verbose=True
    )

    # LLMから応答を予測
    response = chain.predict(input_message=input_message)
    return response

# --- StreamlitアプリケーションのUI ---
st.title("Lesson-21-Assignment：専門家LLMとの対話アプリケーション")

st.write("##### パターン１: スポーツトレーナーとの対話")
st.write("トレーニングメニューの相談や、食事のアドバイスを受けることができます。")
st.write("##### パターン２: 医師との対話")
st.write("体調や症状から病気の可能性を相談したり、健康診断の結果について質問することができます。")

selected_item = st.radio(
    "対話する専門家を選択してください。",
    ["スポーツトレーナー", "医師"]
)

st.divider()

# 選択された専門家に応じた会話履歴と入力フィールドの表示
st.write("##### 会話履歴:")

# 現在選択されている専門家に対応するメモリを取得
current_display_memory = None
if selected_item == "スポーツトレーナー":
    current_display_memory = st.session_state.trainer_memory
elif selected_item == "医師":
    current_display_memory = st.session_state.doctor_memory

# 会話履歴を画面に表示 (st.chat_message を使用してUIを改善)
if current_display_memory:
    # メモリ内のメッセージをループして表示
    for message in current_display_memory.buffer_as_messages:
        if isinstance(message, HumanMessage):
            # ユーザーのメッセージは 'user' アバターで表示
            with st.chat_message("user"):
                st.write(message.content) # ここでメッセージ内容のみ表示
        elif isinstance(message, AIMessage):
            # AIのメッセージは 'assistant' アバターで表示
            with st.chat_message("assistant"):
                st.write(message.content) # ここでメッセージ内容のみ表示
else:
    st.write("専門家を選択してください。")

# 入力フィールドと送信ボタン
# keyをselected_itemに応じて変更することで、切り替え時に内容がクリアされないようにする
input_label = ""
input_placeholder = ""
if selected_item == "スポーツトレーナー":
    input_label = "トレーニングや食事に関する質問を入力してください。"
    input_placeholder = "例: 下半身の筋肉をバランスよくトレーニングするためのメニューは？"
elif selected_item == "医師":
    input_label = "体調や症状に関する質問を入力してください。"
    input_placeholder = "例: 最近頭痛と吐き気がひどいのですが、何か病気の可能性はありますか？"

# テキスト入力と送信ボタンをフォームで囲むことで、入力後に自動的にクリアされる
with st.form(key=f"chat_form_{selected_item}", clear_on_submit=True):
    user_input = st.text_input(
        label=input_label,
        placeholder=input_placeholder,
        key=f"input_text_{selected_item}" # 同じinput_text変数でも、キーを分けることで入力内容が保持される
    )
    submit_button = st.form_submit_button(label="送信")

# 送信ボタンが押され、かつユーザー入力がある場合のみ処理を実行
if submit_button and user_input:
    # ユーザーの入力があった場合のみAI応答を生成
    with st.spinner("AIが回答を生成中です..."): # AI応答中にスピナーを表示
        response = unified_prompt(user_input, selected_item)
    
    # 応答後、Streamlitを再実行して最新の履歴を反映させる
    st.rerun()

