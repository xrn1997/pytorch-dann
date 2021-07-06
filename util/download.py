from urllib.request import urlretrieve
import os
import time

import requests


def download(url, path):
    """
    下载github文件。
    代码存在问题，需要修复。

    :param url: download address
    :param path: storage path

    """
    if not os.path.exists(path):  # 看是否有该文件夹，没有则创建文件夹
        os.mkdir(path)
    try:
        start = time.time()  # 下载开始时间
        html = requests.get(url, stream=True)  # stream=True必须写上
        size = 0  # 初始化已下载大小
        chunk_size = 1024  # 每次下载的数据大小
        content_size = int(html.content)  # 下载文件总大小
        if html.status_code == 200:  # 判断是否响应成功
            print('Start download,[File size]:{size:.2f} MB'.format(
                size=content_size / chunk_size / 1024))  # 开始下载，显示下载文件大小
            filepath = path + '\name.extension name'  # 设置图片name，注：必须加上扩展名
            with open(filepath, 'wb') as file:  # 显示进度条
                for temp in html.iter_content(chunk_size=chunk_size):
                    file.write(temp)
                    size += len(temp)
                    print('\r' + '[下载进度]:%s%.2f%%' % (
                        '>' * int(size * 50 / content_size), float(size / content_size * 100)), end=' ')
        end = time.time()  # 下载结束时间
        print('Download completed!,times: %.2f秒' % (end - start))  # 输出下载用时时间
    except requests.exceptions.Timeout as e:
        print('请求超时' + str(e))
    except requests.exceptions.HTTPError as e:
        print('http请求错误：' + str(e))
    else:
        print("下载成功")
    return

