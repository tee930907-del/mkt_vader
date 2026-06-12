# -*- coding: utf-8 -*-
"""
리뷰 워드 클라우드 + AI 마케팅 인사이트
실행: py -3.13 -m streamlit run review_wordcloud_app.py
"""
import io, re, textwrap, random
from collections import Counter
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from kiwipiepy import Kiwi
import google.generativeai as genai
kiwi = Kiwi()
STOPWORDS = {
    "것","수","등","더","위","때","중","곳","걸","뭐","좀","잘",
    "그","이","저","또","및","를","에","의","가","은","는","로",
    "와","과","도","들","나","때문","거","게","데","점","듯",
    "정도","제","내","네","님","분","개","번","말","다른","다시",
    "하나","여기","거기","어디","언제","어떻게","왜","무엇","누구",
    "자체","사용","구매","구입","주문","배송","제품","상품","후기",
    "계속","부분","정말","진짜","완전","너무","아주","매우","엄청",
    "생각","느낌","기분","처음","마음",
}
POSITIVE_WORDS = {"좋다","좋아","좋은","최고","만족","추천","훌륭","완벽","좋았","대박","사랑","예쁘","깔끔","부드럽","촉촉","보습","향기","고급","산뜻","개운","효과","굿","괜찮","편하","맘에"}
NEGATIVE_WORDS = {"별로","싫","나쁘","아쉽","불만","최악","후회","실망","짜증","자극","따갑","건조","거칠","아프","불편","비싸","냄새","가렵","트러블","뾰루지","알레르기","안좋","못쓰"}
def extract_nouns(text):
    result = kiwi.analyze(text)
    return [t.form for t in result[0][0] if t.tag in ("NNG","NNP","SL") and len(t.form)>=2 and t.form not in STOPWORDS]
def classify_sentiment_by_text(text):
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    return "positive" if pos>neg else ("negative" if neg>pos else "neutral")
import os, subprocess, glob
def get_korean_font_path():
    # 1. Streamlit Cloud (Linux) — 다양한 Nanum 폰트 경로 탐색
    linux_nanum_patterns = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/*.ttf",
        "/usr/share/fonts/opentype/nanum/*.ttf",
        "/usr/share/fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/NanumGothic.ttf",
    ]
    for pattern in linux_nanum_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # 2. Linux — fc-list로 한글 폰트 찾기
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko", "file"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            font_path = result.stdout.strip().split("\n")[0].split(":")[0].strip()
            if os.path.exists(font_path):
                return font_path
    except Exception:
        pass

    # 3. Linux — find로 Nanum 폰트 검색
    try:
        result = subprocess.run(
            ["find", "/usr/share/fonts", "-name", "Nanum*", "-type", "f"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            font_path = result.stdout.strip().split("\n")[0].strip()
            if os.path.exists(font_path):
                return font_path
    except Exception:
        pass

    # 4. Windows 폰트 경로
    for p in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"]:
        if os.path.exists(p): 
            return p
            
    # 5. font_manager로 설치된 폰트 탐색
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ["malgun", "gulim", "nanum", "gothic"]): 
            return f.fname
            
    return ""

# matplotlib 폰트 캐시 재빌드 (Streamlit Cloud에서 필요)
if os.path.exists("/usr/share/fonts"):
    fm._load_fontmanager(try_read_cache=False)

KOREAN_FONT = get_korean_font_path()

REVIEW_COLS = ["리뷰","내용","리뷰내용","리뷰 내용","리뷰 본문","리뷰본문","후기","review","content","text","comment","body"]
RATING_COLS = ["별점","평점","점수","rating","score","star","stars"]
def find_col(df, candidates):
    m = {c.strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in m: return m[c]
    return None
def make_wc(freq, cmap="Set2"):
    if not freq:
        fig,ax=plt.subplots(figsize=(12,5)); ax.text(.5,.5,"데이터 없음",ha="center",va="center",fontsize=24,color="#666",transform=ax.transAxes); ax.set_facecolor("#0e1117"); fig.patch.set_facecolor("#0e1117"); ax.axis("off"); return fig
    wc=WordCloud(font_path=KOREAN_FONT or None,width=1200,height=600,background_color="#0e1117",colormap=cmap,max_words=80,prefer_horizontal=.7,min_font_size=14,max_font_size=120).generate_from_frequencies(freq)
    fig,ax=plt.subplots(figsize=(12,5)); ax.imshow(wc,interpolation="bilinear"); ax.axis("off"); fig.patch.set_facecolor("#0e1117"); plt.tight_layout(pad=0); return fig
def fig_bytes(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#0e1117",edgecolor="none"); buf.seek(0); return buf.getvalue()
def show_kw_table(freq, top_n, label):
    items=freq.most_common(top_n)
    if not items: st.info(f"{label} 키워드 없음"); return
    tdf=pd.DataFrame(items,columns=["키워드","빈도"]); tdf.index=range(1,len(tdf)+1); tdf.index.name="순위"
    st.dataframe(tdf,use_container_width=True,height=min(38*len(tdf)+38,600))
    st.download_button(f"📥 {label} CSV",tdf.to_csv(encoding="utf-8-sig"),f"{label}_keywords.csv","text/csv",use_container_width=True,key=f"dl_{label}")
def build_prompt(pos_kw, neg_kw, neg_samples, total, pos_n, neg_n):
    pk=", ".join(f"{w}({c})" for w,c in pos_kw[:15])
    nk=", ".join(f"{w}({c})" for w,c in neg_kw[:15])
    ns="\n".join(f"- {r[:200]}" for r in neg_samples[:15])
    return textwrap.dedent(f"""\
    당신은 전천후 데이터 사이언티스트이자 마케팅 전략가입니다.
    ## 분석 데이터
    - 전체 리뷰: {total:,}개 / 긍정: {pos_n:,}개 / 부정: {neg_n:,}개
    - 긍정 키워드 TOP15: {pk}
    - 부정 키워드 TOP15: {nk}
    ## 부정 리뷰 샘플
    {ns}
    위 데이터 기반으로 아래 4단계 마케팅 보고서를 한국어 마크다운으로 작성하세요.
    # Step 1: 데이터 탐색 및 카테고리 정의
    1. 제품/서비스 카테고리 정의
    2. 핵심 키워드 TOP10 → [긍정적 특징(USP)] vs [부정적 불만(Pain Point)] 분류
    # Step 2: 동적 마케팅 인사이트 도출 (3가지 전략)
    1. [강점 극대화] 긍정 키워드 → 광고 카피 / 메인 후킹 문구 제안
    2. [위기 및 이탈 방지] 부정 키워드 → 상세페이지 해명·보완 전략
    3. [사용 맥락 분석(TPO)] 사용 상황 분석 → 타겟 마케팅 방향
    # Step 3: 취약 지점 심층 분석 (Voice of Customer)
    1. 문제 키워드 2~3개 선정
    2. 위 리뷰 원문 인용하며 구체적 불만 분석
    3. 마케터가 놓치기 쉬운 디테일한 불만 포인트 요약
    # Step 4: 실행 가능한 액션 플랜
    - 즉시 실행 가능한 광고 소재 아이디어 3가지 (이미지/영상 컨셉 + 광고 카피)
    """)
# ============================================================
st.set_page_config(page_title="리뷰 워드 클라우드 + AI 인사이트", page_icon="☁️", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700;900&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif}
.main-header{text-align:center;padding:2rem 0 .5rem}
.main-header h1{font-size:2.6rem;font-weight:900;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.main-header p{color:#9ca3af;font-size:1.05rem}
.sc{border-radius:14px;padding:1.2rem;text-align:center;border:1px solid rgba(99,102,241,.25)}
.sc.p{background:linear-gradient(135deg,#064e3b,#065f46)}.sc.n{background:linear-gradient(135deg,#4c0519,#881337)}.sc.a{background:linear-gradient(135deg,#1e1b4b,#312e81)}
.sn{font-size:2rem;font-weight:900;color:#e0e7ff}.sl{color:#9ca3af;font-size:.85rem;margin-top:.2rem}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>☁️ 리뷰 워드 클라우드 + AI 인사이트</h1><p>리뷰 파일 업로드 → 긍정·부정 워드 클라우드 → AI 마케팅 보고서</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("API Key", type="password", help="https://aistudio.google.com/apikey")
    st.markdown("---")
    st.markdown("### ⚙️ 설정")
    max_words = st.slider("최대 단어 수",30,200,80,step=10)
    min_wl = st.slider("최소 글자 수",1,5,2)
    top_n = st.slider("상위 키워드 수",10,50,20,step=5)
    st.markdown("---")
    st.markdown("### 🚫 추가 불용어")
    csw = st.text_area("제외 단어 (쉼표 구분)", placeholder="배송, 주문")
uploaded = st.file_uploader("리뷰 파일 업로드 (.xlsx, .csv)", type=["xlsx","csv"])
if uploaded:
    try:
        if uploaded.name.endswith(".xlsx"): df=pd.read_excel(uploaded,engine="openpyxl")
        else:
            raw=uploaded.read()
            for enc in ["utf-8-sig","utf-8","cp949","euc-kr"]:
                try: td=raw.decode(enc); break
                except: continue
            else: td=raw.decode("utf-8",errors="replace")
            df=pd.read_csv(io.StringIO(td))
    except Exception as e: st.error(f"파일 오류: {e}"); st.stop()
    st.success(f"✅ **{uploaded.name}** — {len(df):,}행")
    rc=find_col(df,REVIEW_COLS); rtc=find_col(df,RATING_COLS)
    if not rc: rc=st.selectbox("리뷰 컬럼 선택:",df.columns.tolist())
    else: st.info(f"📌 리뷰: **{rc}**"+(f" | 별점: **{rtc}**" if rtc else ""))
    if rtc:
        with st.sidebar:
            st.markdown("---"); st.markdown("### ⭐ 별점 기준")
            pmin=st.slider("긍정 최소",1,5,4); nmax=st.slider("부정 최대",1,5,2)
    esw={w.strip() for w in csw.split(",") if w.strip()} if csw else set()
    if st.button("🚀 워드 클라우드 + 인사이트 생성",type="primary",use_container_width=True):
        rdf=df[[rc]].copy(); rdf["text"]=rdf[rc].fillna("").astype(str)
        if rtc:
            df[rtc]=pd.to_numeric(df[rtc],errors="coerce"); rdf["rating"]=df[rtc]
            rdf["sent"]=rdf["rating"].apply(lambda r: "positive" if r>=pmin else ("negative" if r<=nmax else "neutral"))
        else: rdf["sent"]=rdf["text"].apply(classify_sentiment_by_text)
        pos_rv=rdf[rdf["sent"]=="positive"]["text"].tolist()
        neg_rv=rdf[rdf["sent"]=="negative"]["text"].tolist()
        all_rv=rdf["text"].tolist()
        def extr(rvs):
            ns=[]
            for r in rvs:
                c=re.sub(r"[^\w\s가-힣a-zA-Z]"," ",r)
                ns.extend(n for n in extract_nouns(c) if len(n)>=min_wl and n not in esw)
            return Counter(ns)
        prog=st.progress(0,"분석 중...")
        prog.progress(10,"긍정 분석..."); pf=extr(pos_rv)
        prog.progress(50,"부정 분석..."); nf=extr(neg_rv)
        prog.progress(90,"전체 분석..."); af=extr(all_rv)
        prog.empty()
        c1,c2,c3=st.columns(3)
        c1.markdown(f'<div class="sc a"><div class="sn">{len(all_rv):,}</div><div class="sl">전체</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="sc p"><div class="sn">{len(pos_rv):,}</div><div class="sl">긍정</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="sc n"><div class="sn">{len(neg_rv):,}</div><div class="sl">부정</div></div>',unsafe_allow_html=True)
        st.markdown("---")
        tw,ti=st.tabs(["☁️ 워드 클라우드","🧠 AI 마케팅 인사이트"])
        with tw:
            st.markdown("## 😊 긍정 워드 클라우드")
            fp=make_wc(dict(pf.most_common(max_words)),"winter"); st.pyplot(fp,use_container_width=True)
            st.download_button("📥 긍정 PNG",fig_bytes(fp),"pos_wc.png","image/png",use_container_width=True,key="dp"); plt.close(fp)
            with st.expander("📊 긍정 키워드",expanded=False): show_kw_table(pf,top_n,"긍정")
            st.markdown("---")
            st.markdown("## 😠 부정 워드 클라우드")
            fn=make_wc(dict(nf.most_common(max_words)),"autumn"); st.pyplot(fn,use_container_width=True)
            st.download_button("📥 부정 PNG",fig_bytes(fn),"neg_wc.png","image/png",use_container_width=True,key="dn"); plt.close(fn)
            with st.expander("📊 부정 키워드",expanded=False): show_kw_table(nf,top_n,"부정")
            st.markdown("---")
            st.markdown("## 🌐 전체 워드 클라우드")
            fa=make_wc(dict(af.most_common(max_words)),"Set2"); st.pyplot(fa,use_container_width=True)
            st.download_button("📥 전체 PNG",fig_bytes(fa),"all_wc.png","image/png",use_container_width=True,key="da"); plt.close(fa)
            with st.expander("📊 전체 키워드",expanded=False): show_kw_table(af,top_n,"전체")
        with ti:
            if not api_key:
                st.warning("⚠️ 사이드바에서 **Gemini API Key**를 입력해주세요.")
                st.markdown("> 🔑 [Google AI Studio](https://aistudio.google.com/apikey)에서 무료 발급 가능")
            else:
                st.markdown("## 🧠 AI 마케팅 인사이트 보고서")
                sn=neg_rv[:15] if len(neg_rv)<=15 else random.sample(neg_rv,15)
                prompt=build_prompt(pf.most_common(15),nf.most_common(15),sn,len(all_rv),len(pos_rv),len(neg_rv))
                with st.spinner("🤖 Gemini AI 분석 중..."):
                    try:
                        genai.configure(api_key=api_key)
                        model=genai.GenerativeModel("gemini-2.5-flash")
                        resp=model.generate_content(prompt)
                        st.markdown(resp.text)
                        st.download_button("📥 보고서 다운로드 (.md)",resp.text.encode("utf-8"),"marketing_insight.md","text/markdown",use_container_width=True,key="dl_rpt")
                    except Exception as e:
                        st.error(f"❌ 오류: {e}")
                        st.info("API Key를 확인해주세요.")