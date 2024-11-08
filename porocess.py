import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 设置相关的文件路径
BG_IMAGE_PATH = "pic/jd_logo.png"  # 背景图片路径
TEXT_PATH = 'call_text.txt'  # 文本文件路径
FONT_PATH = 'msyh.ttf'  # 字体文件路径
STOPWORDS_PATH = 'stopword.txt'  # 停用词文件路径


def load_stopwords(filepath):
    """加载停用词"""
    with open(filepath, encoding='utf-8') as f:
        return f.read().split('\n')
        # return set(f.read().splitlines())


def clean_text(text, stopwords):
    """去除停用词"""
    words = jieba.cut(text, cut_all=False)
    return ''.join(word for word in words if word.strip() not in stopwords and len(word.strip()) > 1)


def preprocess_text(filepath, stopwords):
    """预处理文本"""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    return clean_text(content, stopwords)

'''
allow_pos:
n: 普通名词 ns: 地名 nt: 机构团体名 nw: 工作职务名 nz: 其他专名 v: 动词 va: 形容词性副词
vc: 副词性连词 ve: 副词性助词 a: 形容词 d: 副词 m: 数量词 q: 量词 r: 代词 s: 处所词 t: 时间词 u: 助词 vx: 形容词性副词 x: 其他
'''
def extract_keywords(text, top_k=1000, allow_pos=('nr',)):
    """提取关键词"""
    tags = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
    return {word: weight for word, weight in tags}


def draw_wordcloud(keywords, font_path, bg_image_path, output_path="wordcloud2.jpg"):
    """生成并显示词云"""
    back_coloring = plt.imread(bg_image_path)
    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        max_words=2000,
        mask=back_coloring
    )
    wc.generate_from_frequencies(keywords)

    plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    wc.to_file(output_path)


def main():
    """主函数"""
    stopwords = load_stopwords(STOPWORDS_PATH)
    processed_text = preprocess_text(TEXT_PATH, stopwords)
    keywords = extract_keywords(processed_text)
    draw_wordcloud(keywords, FONT_PATH, BG_IMAGE_PATH)


if __name__ == '__main__':
    main()
