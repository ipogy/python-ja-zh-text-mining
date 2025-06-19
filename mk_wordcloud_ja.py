import pandas as pd
import MeCab
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
    CSVファイルから日本語テキストを抽出し、単語の出現頻度を計算する関数。
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

    # 'lang'が'ja'のテキストを抽出
    japanese_texts = df[df['lang'] == 'ja']['text'].dropna()

    return japanese_texts


def get_word_count(japanese_texts):
    # MeCabの初期化（品詞情報を取得するため、-Owakati は使用しない）
    mecab = MeCab.Tagger()

    # ストップワードの定義
    stop_words = [
        'する', 'いる', 'こと', 'もの', '的', 'よう', 'なる', 'ある', 'ない',
        'これ', 'それ', 'あれ', 'ため', '提供', '展開', 'ユーザー',
        'いる', 'なる', 'ある', 'れる', 'られる', 'せる', 'させる',
        'いう', 'おる', 'くださる', 'さ', 'し', 'ん',
        'だ', 'です', 'ます', 'である', 'であった', 'ですます', 'でしょう', 'だろう',
        # 助詞や接続詞は、以下で定義する品詞除外リストと重複するため、
        # 個別の単語としてここで定義する必要性は低くなります。
        # ただし、特定の助詞などを個別にストップワードとして残したい場合は維持してください。
        # 例えば、「の」という単語自体が特別な意味を持つ文脈では残すなど。
        # 現状のリストは保持します。
        'の', 'が', 'と', 'に', 'を', 'へ', 'で', 'から', 'まで', 'より', 'も', 'な', 'ば', 'か', 'ね', 'よ', 'ねば', 'まい',
        'そして', 'しかし', 'ただし', 'また', 'または', 'あるいは', 'なので', 'つまり', '例えば', '一方', 'なぜなら', '故に',
        'この', 'その', 'あの', 'どの', 'こちら', 'そちら', 'あちら', 'どちら', 'ここ', 'そこ', 'あそこ', 'どこ',
        'いつ', 'どこ', 'だれ', '何', 'どう', 'なぜ',
        'はい', 'いいえ', 'ありがとう', 'すみません', 'ごめんなさい', 'こんにちは', 'こんばんは',
        'ますます', 'どんどん', 'かなり', 'とても', '非常に', '少し', 'もっと', 'あまり', '全く',
        '一般', '具体', '全て', '一部', '全体', '各種', '各々', 'それぞれ', '多く', '少なく',
        '中', '外', '上', '下', '前', '後', '右', '左', '間', '内', '辺り',
        '場合', '時', '時点', '間', '際', '頃', '中', '後', '前', '度',
        '〇〇', '☓☓', 'ああ', 'うん', 'ええ', 'おや', 'おい',
        'さん', '様', '殿', '氏', '君', 'ちゃん',
        'そして', 'それから', 'その後', 'さらに', '加えて', '一方で',
        'いわゆる', 'いわば', '～において', '～に対して', '～にとって', '～による', '～に伴い',
        'など', '等', 'ばかり', 'ほど', 'くらい', 'さえ', 'こそ', 'しか', 'すら', 'ずつ', 'だけ',
        'ようとする', 'ことである', 'ものである', 'ことになる', 'もと', 'ほか', 'ゆえ', 'ゆえに', 'うえ',
        'できる', 'できない', '可能', '不可能', '必要', '不要', '重要', '不可欠', '必須',
        '点', '面', '際', '通り', '形', '種類', '例', 'ケース', '場合', 'ところ',
        '及び', '並びに', '或いは', '且つ', '但し', '然し',
        '上記', '下記', '当該', '弊社', '貴社', '当社',
        '例えば', '特に', '主に', '特に', 'さらに', 'なお', 'さて',
        'て', 'で',
        'これら', 'それら',
        '〜という', '〜といった', '〜と考える', '〜と思われる', '〜と考えられる',
        '弊社', '貴社', '御社', '当社', '自社', '他社',
        '〜ます', '〜です',
        'ん', 'さ', 'ね', 'な', 'わ',
    ]

    # 除外する品詞のリスト（IPA辞書の場合）
    exclude_hinshi = [
        '助詞',
        '助動詞',
        '記号',
        '感動詞',
        '接続詞',
        '接頭詞',
        'フィラー',
        # 必要に応じて追加・削除
        # '名詞,非自立', # 例：「こと」「もの」などが含まれる。ストップワードリストと重複する場合は慎重に
        # '動詞,非自立',
        # '形容詞,非自立',
        # '副詞', # 頻出する副詞はストップワードリストで対応し、残すものも考慮
        # '連体詞', # 特定の品詞（「この」「その」など）はストップワードで対応し、残すものも考慮
    ]

    all_filtered_words = []
    for text in japanese_texts:
        # 半角カナを全角カナに、全角英数を半角英数に変換など
        # 日本語に大文字・小文字の区別はないため、英字がほとんどない場合は無意味になります。
        # 英字も処理対象に含める場合のみ検討。
        text = unicodedata.normalize('NFKC', text)

        # 全て大文字化で統一（英字が含まれる場合のみ有効）
        text = text.upper()

        # URLと記号を除去
        text = re.sub(r'https?://\S+', '', text)
        # 記号除去はMeCabの`記号`品詞で大部分をカバーできるため、必須ではないが、
        # MeCabの処理前に除去することで、余計なノード生成を防ぐ効果はあります。
        text = re.sub(r'[^\w\s]', '', text)

        # 連続する空白を単一の空白に置き換え、前後の空白を除去
        text = re.sub(r'\s+', ' ', text).strip()

        # 空白だけの文字列や非常に短い文字列を除外
        if not text:
            continue

        # 形態素解析して単語リストに追加
        node = mecab.parseToNode(text)
        while node:
            word = node.surface
            # MeCabの解析結果の最初と最後のノードは特殊なためスキップ
            if word == '' or node.feature == 'BOS/EOS':
                node = node.next
                continue

            hinshi = node.feature.split(',')[0] # 品詞を取得

            # 品詞とストップワードによるフィルタリング、および1文字の単語の除去
            # strip()で前後の空白を除去し、空文字でないことを確認
            if (hinshi not in exclude_hinshi and
                word not in stop_words and
                len(word.strip()) > 1): # 1文字の単語をここで除去
                all_filtered_words.append(word)

            node = node.next

    # 単語の出現頻度を計算
    word_count = Counter(all_filtered_words)
    return word_count


def create_wordcloud(word_count, title='Word Cloud'):
    """
    単語の出現頻度からワードクラウドを生成する関数（ユーザー提供のコードをベース）
    """
    if not word_count:
        print("WordCloudを生成するための単語がありません。")
        return

    # ★★★ ご利用の環境に合わせて日本語フォントのパスを指定してください ★★★
    # macOS の例: font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
    # Windows の例: font_path = 'C:/Windows/Fonts/YuGothM.ttc'
    # Google Colab/Linux の例: font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    # font_path = '/usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf'

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
        print("ご利用のOSに合わせて 'font_path' を正しい日本語フォントのパスに修正してください。")
        return

    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud from CSV (Japanese Text)')

    output_directory = 'output/ja'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plt.savefig(f'{output_directory}/{title.replace(" ", "_")}'+'_ja'+'.png', dpi=800)
    print(f"\nWordCloudを '{output_directory}/{title.replace(' ', '_')}_ja.png' として保存しました。")

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
        create_wordcloud(word_count_result, title="wordcloud")

    try:
        df = pd.read_csv(csv_path)
        # df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_path}' が見つかりません。")

    # 日本語テキストのみを対象
    df_ja = df[df['lang'] == 'ja'].copy()

    # --- 2. 感情分析の準備 ---
    print("感情分析モデルを読み込んでいます...（初回は時間がかかります）")
    # 日本語の感情分析モデルを指定して、パイプラインを作成
    sentiment_analyzer = pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")
    print("モデルの読み込みが完了しました。")

    # --- 3. 感情分析の実行 ---
    # テキストが空や欠損している場合のエラーを避ける
    texts_to_analyze = df_ja['text'].dropna().tolist()
    if not texts_to_analyze:
        print("分析対象の日本語テキストがありません。")

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
    df_ja['sentiment'] = sentiments

    # --- 4. 結果の表示 ---
    # print("\n--- 感情分析結果 ---")
    # print(df_ja[['text', 'sentiment']])

    print("\n--- 感情の集計 ---")
    print(df_ja['sentiment'].value_counts())

    # --- 5. ポジティブ/ネガティブ別のWordCloud作成 ---
    positive_texts = df_ja[df_ja['sentiment'] == 'POSITIVE']['text'].tolist()
    negative_texts = df_ja[df_ja['sentiment'] == 'NEGATIVE']['text'].tolist()

    # ポジティブなWordCloud
    positive_word_count = get_word_count(positive_texts)
    create_wordcloud(positive_word_count, title='Positive Words')

    # ネガティブなWordCloud
    negative_word_count = get_word_count(negative_texts)
    create_wordcloud(negative_word_count, title='Negative Words')

    print("\n--- 全ての分析が完了しました ---")