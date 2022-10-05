# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:13:25 2022

@author: hussain
"""

# pip install pytube
from pytube import YouTube
url = 'https://www.youtube.com/watch?v=gQ3IgGzrN9Q&ab_channel=FahadAbbasi'
output_path = ''
# YouTube(url).streams.first().download(output_path)     # lowest quality
YouTube(url).streams.get_highest_resolution().download(output_path)       # highest quality
