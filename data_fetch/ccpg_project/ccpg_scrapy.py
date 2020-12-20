import time
import math
import requests
from lxml import etree

# 每页的条目数
ITEM_PER_PAGE = 20

# 竞标类型信息
BID_TYPE_DICT = {
    1: "公开招标", 2: "询价公告", 3: "竞争性谈判", 4: "单一来源",
    5: "资格预审", 6: "邀请公告", 7: "中标公告", 8: "更正公告",
    9: "其他公告", 10: "竞争性磋商", 11: "成交公告", 12: "终止公告"
}

# 品目信息
PIN_MU_DICT = {1: "货物类", 2: "工程类", 3: "服务类"}

# 竞标分类信息
BID_SORT_DICT = {1: "中央公告", 2: "地方公告"}


def build_parameters_dict(start_time: str, end_time: str) -> dict:
    """
    构造parameter字典，用于get请求。
    :param start_time:
    :param end_time:
    :return:
    """
    parameters = {'searchtype': '2', 'start_time': start_time, 'end_time': end_time, 'timeType': '6'}
    return parameters


def get_initial_html(url: str, headers: dict, params: dict) -> str:
    """
    获得网页源码信息。
    :return:
    """
    try:
        response = requests.get(url, headers=headers, params=params)
        print(response.url)
        if response.status_code == 200:
            html = response.content.decode('utf-8', 'ignore').replace(u'\xa9', u'')
            return html
        else:
            print(response.status_code)
    except requests.ConnectionError:
        return "Fail"


def calculate_pages_number(html: str) -> int:
    """
    获取每个查询结果显示的数据量，然后通过每页20条的数据转换成共计多少页。
    :return:
    """
    html = etree.HTML(html)
    num_list = html.xpath('/html/body/div[5]/div[1]/div/p/span[2]/text()')
    num = int(num_list[0])
    pages_number = math.ceil(num / ITEM_PER_PAGE)
    return pages_number


def get_li_tags(html: str) -> list:
    """
    定位到每条数据的li标签，方便后面根据li标签进行遍历，提取信息。
    :return:
    """
    html = etree.HTML(html)
    li_tag_list = html.xpath("/html/body/div[5]/div[2]/div/div/div[1]/ul/li")
    return li_tag_list


def sparse_li2a(li_tag_list: list) -> tuple:
    """
    定位li标签下的a标签，提取标书文件的标题和url。
    :param li_tag_list:
    :return:
    """
    href_list = list()
    title_list = list()
    for li in li_tag_list:
        href = li.xpath("a/@href")
        add_href = filter_space(href)[0]
        title = li.xpath("a/text()")
        print("*****" * 10)
        print(title)
        href_list.append(add_href)
        add_title = filter_space(title)[0]
        print("$$$$$" * 10)
        print(title)
        title_list.append(add_title)
    return href_list, title_list


def sparse_li2span(li_tag_list: list) -> tuple:
    """
    定位li标签下的span标签，提取标书的概要信息。
    :param li_tag_list:
    :return:
    """
    time_list = list()
    purchase_list = list()
    agency_list = list()
    area_list = list()
    for li in li_tag_list:
        buyer = li.xpath("span/text()")
        add_buyer = filter_space(buyer)
        if len(add_buyer) == 3:
            add_time = add_buyer[0]
            time_list.append(add_time)
            add_purchase = add_buyer[1]
            purchase_list.append(add_purchase)
            add_agency = add_buyer[2]
            agency_list.append(add_agency)
        else:
            time_list.append("no time")
            purchase_list.append("no purchase")
            agency_list.append("no agency")

        # if len(add_buyer) != 3:
        #     continue
        # else:
        #     add_time = add_buyer[0]
        #     time_list.append(add_time)
        #     add_purchase = add_buyer[1]
        #     purchase_list.append(add_purchase)
        #     add_agency = add_buyer[2]
        #     agency_list.append(add_agency)

        area = li.xpath("span/a/text()")
        add_area = filter_space(area)
        if not add_area:
            add_area.append("no data")
        area_list.append(add_area[0])
    return time_list, purchase_list, agency_list, area_list


def filter_space(xpath_result: list) -> list:
    """
    除去xpath提取结果中包含的空格符，换行符，制表符等
    :param xpath_result:
    :return:
    """
    return_list = list()
    temp_list = " ".join(xpath_result).strip().split("|")
    for element in temp_list:
        temp_element = element.replace("\n", " ").replace("\r", " ").strip()
        if temp_element:
            return_list.append(temp_element)
    return return_list


def main():

    initial_url = 'http://search.ccgp.gov.cn/bxsearch?'

    headers = {
        'Cookie': 'JSESSIONID=EgPd86-6id_etA2QDV31Kks3FrNs-4gwHMoSmEZvnEktWIakHbV3!354619916; '
                  'Hm_lvt_9f8bda7a6bb3d1d7a9c7196bfed609b5=1602214804; '
                  'Hm_lpvt_9f8bda7a6bb3d1d7a9c7196bfed609b5=1602214892; '
                  'JSESSIONID=OBoLczbR_k89lC8sOuKF4W-46DVqKEd5u7isUpSyOjE6D0nBP94c!1675672049; '
                  'Hm_lvt_9459d8c503dd3c37b526898ff5aacadd=1602214902,1602214928,1602214932,1602214937; '
                  'Hm_lpvt_9459d8c503dd3c37b526898ff5aacadd=1602214937',
        'Host': 'search.ccgp.gov.cn',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/85.0.4183.121 Safari/537.36 '
    }

    start_time = "2014:04:01"
    end_time = "2014:04:30"
    count = 0

    for pin_mu_index in range(1, len(PIN_MU_DICT) + 1):
        # 后面写文件时需要此字段，分类标签，共计三种，下面的代码分别实现。
        pin_mu_type = PIN_MU_DICT.get(pin_mu_index)
        output_file_path = start_time.split(":")[0] + start_time.split(":")[1] + "save_data.txt"
        for bid_type_index in range(9, 10):
            bid_type_type = BID_TYPE_DICT.get(bid_type_index)
            for bid_sort_index in range(1, len(BID_SORT_DICT) + 1):
                bid_sort_type = BID_SORT_DICT.get(bid_sort_index)
                # 第一步构造标签，目的是需要找到特定参数下，共计多少页，在这里计算pages number。
                params = build_parameters_dict(start_time, end_time)
                params['bidSort'] = str(bid_sort_index)
                params['pinMu'] = str(pin_mu_index)
                params['bidType'] = str(bid_type_index)
                time.sleep(5)
                response = get_initial_html(url=initial_url, headers=headers, params=params)
                pages_number = calculate_pages_number(response)
                for page_index in range(107, pages_number + 1):
                    params['page_index'] = page_index
                    time.sleep(6)
                    response = get_initial_html(url=initial_url, headers=headers, params=params)
                    li_tags_list = get_li_tags(response)
                    href_list, title_list = sparse_li2a(li_tags_list)
                    time_list, purchase_list, agency_list, area_list = sparse_li2span(li_tags_list)

                    with open(output_file_path, mode="a+", encoding="utf-8") as fw:
                        for num in range(len(href_list)):
                            fw.write(bid_type_type + "\t" + bid_sort_type + "\t" + pin_mu_type + "\t" +
                                     title_list[num] + "\t" + time_list[num] + "\t" + purchase_list[num] + "\t" +
                                     agency_list[num] + "\t" + area_list[num] + "\t" + href_list[num] + "\n")
                    count += 1
                    print("finish writing %d pages" % count)
                print(pin_mu_type + " " + bid_type_type + " " + "finish !")


if __name__ == '__main__':
    main()
