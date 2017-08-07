#! -*- coding: utf-8 -*-

import os
import re
import csv
import time
import urllib2
from bs4 import BeautifulSoup

url_template = "http://actress.dmm.co.jp/-/list/=/keyword=%s/page=%d/"
header_template = re.compile(u"[0-9]+人 - 全(\d+)ページ中 [0-9]+ページ目")


def download_image(image_url, local_image_path):
    """
    画像をダウンロードする関数
    """
    print "downloading " + image_url
    local_image_file = open(local_image_path, "wb")
    image_data = urllib2.urlopen(image_url, "html.parser").read()
    local_image_file.write(image_data)
    local_image_file.close();
    time.sleep(1.0) # 10秒待機

if __name__ == "__main__":

    hiragana = ["a", "i", "u", "e", "o",
                "ka", "ki", "ku", "ke", "ko",
                "sa", "si", "su", "se", "so",
                "ta", "ti", "tu", "te", "to",
                "na", "ni",       "ne", "no",
                "ha", "hi", "hu", "he", "ho",
                "ma", "mi", "mu", "me", "mo",
                "ya", "yu", "yo",
                "ra", "ri", "ru", "re", "ro",
                "wa"]

    # CSVを作成する
    image_csv = open("./image.csv", "w")
    csv_writer = csv.writer(image_csv)
    for h in hiragana:
        current_page = 1
        max_page = 10000

        while current_page <= max_page:
            # gets html
            url =  url_template % (h, current_page)
            html = urllib2.urlopen(url).read()
            time.sleep(10.0);
            soup = BeautifulSoup(html, "html.parser")

            if current_page == 1: # 現在の先頭文字に対する最大ページ数を取得する
                td_header = soup.find_all("td", attrs = {"class": "header"})[1]
                # 2つ目の
                td_header.find("br").extract()
                # タグ内にあるタグを削除する（しないとタグ内の文字を取得できない）
                td_header_string = td_header.string
                match = header_template.match(td_header_string)
                max_page = int(match.groups(1)[0])

            # 画像urlのリストを作成する
            td_pics = soup.find_all("td", attrs = {"class": "pic"})
            for td_pic in td_pics:
                image_url = td_pic.find("img").get("src")
                image_name =os.path.basename(image_url)
                actress_name = td_pic.find("img").get("alt").encode("utf-8")
                local_image_path = "./images/" + image_name
                csv_writer.writerow([image_name, image_url, local_image_path, actress_name])
            current_page += 1

    image_csv.close()

    # 画像をダウンロードする
    image_csv = open("./image.csv", "r")
    csv_reader = csv.reader(image_csv)
    for row in csv_reader:
        image_url = row[1]
        local_image_path = row[2]
        download_image(image_url, local_image_path)
    image_csv.close()
