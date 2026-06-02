import csv
import os
import urllib.request
import ssl

#fix for potential ssl certificate errors when downloading
ssl._create_default_https_context = ssl._create_unverified_context

#download model file if not already present
def download_if_missing(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {os.path.basename(filename)}...")
        urllib.request.urlretrieve(url,filename)

