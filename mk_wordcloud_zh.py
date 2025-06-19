import pandas as pd
import jieba.posseg as pseg
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import unicodedata
from transformers import pipeline
from tqdm import tqdm
import os

# ---【事前準備】---
# 必要なライブラリがインストールされていない場合は、以下のコマンドでインストールしてください。
# !pip install pandas mecab-python3 unidic-lite wordcloud matplotlib

def get_word_count_from_csv(csv_path):
    """
    CSVファイルから中国語テキストを抽出し、単語の出現頻度を計算する関数。
    品詞とストップワードによるフィルタリングを含む。
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_path}' が見つかりません。")
        return None
    except Exception as e:
        print(f"エラー: ファイルの読み込み中にエラーが発生しました: {e}")
        return None

    # 'lang'列と'text'列の存在を確認
    if 'lang' not in df.columns or 'text' not in df.columns:
        print("エラー: CSVファイルに 'lang' または 'text' 列が見つかりません。")
        return None

    # 'lang'が'zh'のテキストを抽出
    chinese_texts = df[df['lang'] == 'zh']['text'].dropna()

    return chinese_texts


def get_word_count(chinese_texts):
    # 中国語のストップワードリスト (Chinese stop word list)
    stop_words_zh = [
        '的', '地', '得', '了', '着', '是', '有', '和', '与', '及', '或', '以及', '虽然', '但是',
        '然而', '因此', '所以', '因为', '由于', '如果', '那么', '即使', '也', '都', '再', '更',
        '很', '非常', '十分', '一定', '肯定', '可能', '也许', '还是', '为了', '为了', '对', '对于',
        '从', '从', '到', '向', '往', '在', '上', '下', '里', '外', '前面', '后面', '之间',
        '个', '位', '种', '些', '本', '次', '件', '名', '张', '块', '辆', '艘', '座', '间', '口', '部',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '人家',
        '这', '那', '这儿', '那儿', '这里', '那里', '哪个', '什么', '怎样', '怎么', '多少',
        '年', '月', '日', '天', '时', '分', '秒', '星期', '周', '现在', '过去', '未来',
        '不', '没', '无', '非', '别', '不要', '没有',
        '啊', '呀', '吧', '吗', '呢', '啦', '嘛', '哈', '嘿', '哟', '嗯', '哇',
        '进行', '发展', '实现', '表示', '认为', '加强', '提高', '提供', '包括', '成为',
        '可以', '需要', '重要', '主要', '一个', '一种', '各种', '各类',
        '关于', '通过', '随着', '通过', '根据',
    ]

    # 除外する品詞のリスト (jieba.posseg POS tags to exclude)
    exclude_pos_zh = [
        'u',  # 助词 (particle)
        'c',  # 连词 (conjunction)
        'p',  # 介词 (preposition)
        'e',  # 叹词 (interjection)
        'o',  # 拟声词 (onomatopoeia)
        'x',  # 未知词 (unknown word)
        'q',  # 量词 (measure word)
        'm',  # 数词 (numeral)
        'w',  # 标点符号 (punctuation)
    ]

    all_filtered_words = []
    for text in chinese_texts:
        text = unicodedata.normalize('NFKC', text)
        text = text.upper()

        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            continue

        words_with_pos = pseg.cut(text)

        for word_obj in words_with_pos:
            word = word_obj.word.strip()
            pos = word_obj.flag

            if (word and
                pos not in exclude_pos_zh and
                word not in stop_words_zh and
                len(word) > 0):

                all_filtered_words.append(word)

    word_count = Counter(all_filtered_words)
    return word_count


def create_wordcloud(word_count, title='Word Cloud'):
    """
    単語の出現頻度からワードクラウドを生成する関数（ユーザー提供のコードをベース）
    """
    if not word_count:
        print("WordCloudを生成するための単語がありません。")
        return

    # ★★★ ご利用の環境に合わせて中国語フォントのパスを指定してください ★★★
    # macOS の例: font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
    # Windows の例: font_path = 'C:/Windows/Fonts/YuGothM.ttc'
    # Google Colab/Linux の例: font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_path = '/usr/share/fonts/truetype/arphic/ukai.ttc'

    try:
        wordcloud = WordCloud(
            font_path=font_path,
            width=1920,
            height=1080,
            background_color='white',
            max_words=400,
            collocations=False
        ).generate_from_frequencies(word_count)
    except IOError:
        print(f"エラー: 指定されたフォントパス '{font_path}' が見つかりません。")
        print("ご利用のOSに合わせて 'font_path' を正しい中国語フォントのパスに修正してください。")
        return

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud from CSV (Chinese Text)')

    output_directory = 'output/zh'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plt.savefig(f'{output_directory}/{title.replace(" ", "_")}'+'_zh'+'.png', dpi=800)
    print(f"\nWordCloudを '{output_directory}/{title.replace(' ', '_')}_zh.png' として保存しました。")

    #plt.show()


# --- メイン処理 ---
if __name__ == '__main__':
    # --- 1. データの読み込み ---
    # 作成したサンプルCSVファイル、またはお手持ちのCSVファイルのパスを指定
    CSV_FILE_PATH = 'sample_dataset.csv'
    csv_path = 'for_export_deepseek.csv'

    # CSVから単語の頻度を計算
    words = get_word_count_from_csv(csv_path)
    # words = get_word_count_from_csv(CSV_FILE_PATH)
    word_count_result = get_word_count(words)

    # ワードクラウドを生成
    if word_count_result:
        create_wordcloud(word_count_result)
        
    try:
        df = pd.read_csv(csv_path)
        # df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_path}' が見つかりません。")

    # 中国語テキストのみを対象
    df_zh = df[df['lang'] == 'zh'].copy()

    # --- 2. 感情分析の準備 ---
    print("感情分析モデルを読み込んでいます...（初回は時間がかかります）")
    # 中国語の感情分析モデルを指定して、パイプラインを作成
    sentiment_analyzer = pipeline("sentiment-analysis", model="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
    print("モデルの読み込みが完了しました。")

    # --- 3. 感情分析の実行 ---
    # テキストが空や欠損している場合のエラーを避ける
    texts_to_analyze = df_zh['text'].dropna().tolist()
    if not texts_to_analyze:
        print("分析対象の中国語テキストがありません。")

    print("\n感情分析を実行中...")


    sentiments = []
    # ★★★ tqdmを使用して進捗バーを表示 ★★★
    # desc="..." でプログレスバーの左側に表示するテキストを指定できます
    for text in tqdm(texts_to_analyze, desc="感情分析中"):
        # 1件ずつ分析を実行
        result = sentiment_analyzer(text[:512])
        # 結果のラベルをリストに追加
        sentiments.append(result[0]['label'])

    # 元のデータフレームに結果を新しい列として追加
    df_zh['sentiment'] = sentiments

    # --- 4. 結果の表示 ---
    # print("\n--- 感情分析結果 ---")
    # print(df_zh[['text', 'sentiment']])

    print("\n--- 感情の集計 ---")
    print(df_zh['sentiment'].value_counts())

    # --- 5. ポジティブ/ネガティブ別のWordCloud作成 ---
    positive_texts = df_zh[df_zh['sentiment'] == 'Positive']['text'].tolist()
    negative_texts = df_zh[df_zh['sentiment'] == 'Negative']['text'].tolist()

    # ポジティブなWordCloud
    positive_word_count = get_word_count(positive_texts)
    create_wordcloud(positive_word_count, title='Positive Words')

    # ネガティブなWordCloud
    negative_word_count = get_word_count(negative_texts)
    create_wordcloud(negative_word_count, title='Negative Words')

    print("\n--- 全ての分析が完了しました ---")