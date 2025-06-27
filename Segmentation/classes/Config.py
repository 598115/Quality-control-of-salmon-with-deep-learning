
import os

### Name of data directory ###
data_dir = 'data'
### Tag names in files ###
tag_blod = 'tag-Blod'
tag_melanin = 'tag-Melanin'
tag_rygg = 'tag-Rygg'
tag_buk1 = 'tag-Buk1'
tag_buk2 = 'tag-Buk2'

class Config:
    """
    Configuration class to manage paths and tag names.
    Attributes:
        WORK_DIR (str): Current working directory.
        DATA_DIR (str): Path to the data directory.
        blod_tag (str): Tag for blood.
        melanin_tag (str): Tag for melanin.
        rygg_tag (str): Tag for rygg.
        buk1_tag (str): Tag for buk1.
        buk2_tag (str): Tag for buk2.
    """
    def __init__(self):
        self.WORK_DIR = os.getcwd()
        self.DATA_DIR = os.path.join(self.WORK_DIR, data_dir)
        self.blod_tag = tag_blod
        self.melanin_tag = tag_melanin
        self.rygg_tag = tag_rygg
        self.buk1_tag = tag_buk1
        self.buk2_tag = tag_buk2
   
    def get_data_dir(self):
        return self.DATA_DIR
    def get_blod_tag(self):
        return self.blod_tag
    def get_melanin_tag(self):
        return self.melanin_tag
    def get_rygg_tag(self):
        return self.rygg_tag
    def get_buk1_tag(self):
        return self.buk1_tag
    def get_buk2_tag(self):
        return self.buk2_tag
    


