import requests
from lxml import etree
"""
根据url获取简书的网页内容，并写入txt文件。
"""


def get_initial_html(url: str, headers: dict) -> str:
    """
    获得网页源码信息。
    :return:
    """
    try:
        response = requests.get(url, headers=headers)
        print(response.url)
        if response.status_code == 200:
            html = response.content.decode('utf-8', 'ignore').replace(u'\xa9', u'')
            return html
        else:
            print(response.status_code)
    except requests.ConnectionError:
        return "Fail"


def parse_text_content(html: str) -> list:
    """
    获取每个查询结果显示的数据量，然后通过每页20条的数据转换成共计多少页。
    :return:
    """
    html = etree.HTML(html)
    need_content = html.xpath('//*[@id="__next"]/div[1]/div/div[1]/section[1]/article/p')
    return need_content


def write_content(content_tags: list):
    with open("first.txt", mode="w", encoding="utf-8") as fw:
        for line_content in content_tags:
            if line_content is None:
                continue
            fw.write(str(line_content.text))
            fw.write("\n")


def run():
    url = "https://www.jianshu.com/p/59ce95840e79"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }
    temp = get_initial_html(url, headers)
    res = parse_text_content(temp)
    write_content(res)


if __name__ == '__main__':
    run()
