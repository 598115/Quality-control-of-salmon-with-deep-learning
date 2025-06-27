
import pandas as pd
import glob
import pandas as pd
import numpy as np
from PIL import Image
import os

import torch
from .Config import Config
from collections import defaultdict
import re
from tqdm import tqdm

class DataProcessor:
    """
    A class for processing and managing image data for semantic segmentation tasks.\n Files are automatically gathered from the data directory
    This class handles various data processing tasks including:
    - Converting masks to grayscale
    - Grouping files (segmentation masks, images) by task number
    - Combining segmentation masks
    - Creating and managing CSV files for data organization
    - Creating data augmentations
    Attributes:
        conf (Config): Configuration object containing system paths and settings
        file_paths (list): List of paths to training/validation label files
        test_file_paths (list): List of paths to test set label files
        grouped_files (list): Grouped training/validation files by task
        test_grouped_files (list): Grouped test files by task
    Methods:
        - grayscale(): Convert mask images to grayscale
        - group_files_by_task(): Group files based on task numbers
        - combine_seg_masks(): Combine individual segmentation masks into zone and spot masks
        - extract_filename(path): Extract image filename from path
        - extract_test_filename(path): Extract test image filename from path
        - extract_annotation_id(id, label_type): Get annotation file path for given ID and label type
        - extract_test_annotation_id(id, label_type): Get test annotation file path for given ID and label type
        - file_exists(filepath): Check if file exists at given path
        - create_processed_data_csv(): Create CSV files with processed data information
        - combine_csv(): Combine multiple CSV files into a single file
        - fixTaskNumCollision(img_change_dir, imgs_paths, csv_path): Fix task number collisions in dataset
        - create_augmentations(): Create augmented versions of images and their masks
    Example:
        processor = DataProcessor()
        processor.grayscale()
        processor.combine_seg_masks()
        processor.create_processed_data_csv()
    """

    def __init__(self):
        self.conf = Config()
        print(f"Current working directory: {os.getcwd()}")
        self.file_paths = glob.glob(os.path.join(self.conf.get_data_dir(),'labels/*.png'))
        self.test_file_paths = { "trym" : glob.glob(os.path.join(self.conf.get_data_dir(),'labels/test/trym/*.png')),
                                "brage" : glob.glob(os.path.join(self.conf.get_data_dir(),'labels/test/brage/*.png')),
                                "lars" : glob.glob(os.path.join(self.conf.get_data_dir(),'labels/test/lars/*.png'))
                                }
        self.test_csv_paths = {"trym" : os.path.join(self.conf.get_data_dir(),'csv_input/test_data/test_data_trym.csv'),
                                "brage" : os.path.join(self.conf.get_data_dir(),'csv_input/test_data/test_data_brage.csv'),
                                "lars" : os.path.join(self.conf.get_data_dir(),'csv_input/test_data/test_data_lars.csv')
                                }
        self.file_paths.sort()
        for key, file_paths in self.test_file_paths.items():
            file_paths.sort()
        print(f"Number of train/val label files found: {len(self.file_paths)}")
        num_test_files = sum(len(paths) for paths in self.test_file_paths.values())
        print(f"Number of test set label files found: {num_test_files}")
        self.grouped_files = defaultdict(list)
        self.test_task_groups = {
            "trym": defaultdict(list),
            "brage": defaultdict(list),
            "lars": defaultdict(list)
        }

    # Convert masks to grayscale 
    def grayscale(self):
        """
        Convert segmentation masks to grayscale.
        """
        for png_file in self.file_paths:
            # Open image and convert to grayscale
            with Image.open(png_file) as img:
                gray_img = img.convert('L')
                
                # Save grayscale image, overwriting the original
                gray_img.save(png_file)

        print(f"Converted {len(self.file_paths)} train/val labels to grayscale")

        for key, file_paths in self.test_file_paths.items():
            for png_file in file_paths:
                # Open image and convert to grayscale
                with Image.open(png_file) as img:
                    gray_img = img.convert('L')            
                    # Save grayscale image, overwriting the original
                    gray_img.save(png_file)
            print(f"Converted {len(file_paths)} test labels by rater '{key}' to grayscale")


    def group_files_by_task(self):
        """
        Group files by task number using regex. Each image has multiple segmentation masks belonging to it, one for each class. These are found in Segmentation/data/labels.\n 
        This functions groups the masks belonging to the same image together with the image using their task numbers.
        """
        task_pattern = re.compile(r'task-(\d+)')
        
        for path in self.file_paths:
            # Extract task number using regex
            match = task_pattern.search(path)
            if match:
                task_number = match.group(1)
                self.grouped_files[task_number].append(path)

        for key, paths in self.test_file_paths.items():
            for path in paths:
                match = task_pattern.search(path)
                if match:
                    task_number = match.group(1)
                    self.test_task_groups[key][task_number].append(path)
            print(f"Grouped {len(paths)} test files by rater '{key}' into {len(self.test_task_groups[key])} task groups")
             
        print(f"Grouped {len(self.file_paths)} train/val files into {len(self.grouped_files)} task groups")

        tag_rygg = self.conf.get_rygg_tag()
        tag_buk1 = self.conf.get_buk1_tag()
        tag_buk2 = self.conf.get_buk2_tag()
        
        # Custom sorting key function
        def sort_key(path):
            if tag_rygg in path:
                return 0
            elif tag_buk1 in path:
                return 1
            elif tag_buk2 in path:
                return 2
            return 3
        
        # Sort files within each group
        self.grouped_files = [sorted(files, key=sort_key) for files in self.grouped_files.values()]
        for key, groups in self.test_task_groups.items():
            self.test_task_groups[key] = [sorted(files, key=sort_key) for files in groups.values()]
        
        return self.grouped_files, self.test_task_groups
    
    def combine_seg_masks(self):
        """
        Combine individual single-class segmentation masks into combined multiclass zone and spot masks.\n
        Masks are saved as numpy arrays in the combined_labels directory.
        """
        
        tag_rygg = Config().get_rygg_tag()
        tag_buk1 = Config().get_buk1_tag()
        tag_buk2 = Config().get_buk2_tag()
        tag_blod = Config().get_blod_tag()
        tag_melanin = Config().get_melanin_tag()

        self.group_files_by_task()
        
        for group in self.grouped_files:

            zone_mask = None
            spot_mask = None

            for file in group:

                seg_arr = np.array(Image.open(file))
                
                if zone_mask is None:
                    zone_mask = np.zeros(seg_arr.shape, dtype=np.uint8)
                    spot_mask = np.zeros(seg_arr.shape, dtype=np.uint8)
                
                if tag_rygg in file:    
                    zone_mask[seg_arr > 0] = 1
                elif tag_buk1 in file:             
                    zone_mask[seg_arr > 0] = 2
                elif tag_buk2 in file:             
                    zone_mask[seg_arr > 0] = 3
                elif tag_blod in file:             
                    spot_mask[seg_arr > 0] = 1
                elif tag_melanin in file:            
                    spot_mask[seg_arr > 0] = 2
            
            task_id = os.path.basename(group[0]).split('-')[1]
            zone_output_dir = os.path.join(self.conf.get_data_dir() ,'combined_labels/zone')
            spot_output_dir = os.path.join(self.conf.get_data_dir() ,'combined_labels/spot')
            os.makedirs(zone_output_dir, exist_ok=True)
            os.makedirs(spot_output_dir, exist_ok=True)
            zone_output_filename = f'{ zone_output_dir}/id-{task_id}-combined-zone.npy'
            spot_output_filename = f'{spot_output_dir}/id-{task_id}-combined-spot.npy'
            np.save(zone_output_filename, zone_mask)
            np.save(spot_output_filename, spot_mask)
        
        print(f"Combined {len(self.grouped_files)} train/val task groups into zone and spot masks")

        for key, groups in self.test_task_groups.items():
            for group in groups:
                zone_mask = None
                spot_mask = None

                for file in group:

                    seg_arr = np.array(Image.open(file))
                    
                    if zone_mask is None:
                        zone_mask = np.zeros(seg_arr.shape, dtype=np.uint8)
                        spot_mask = np.zeros(seg_arr.shape, dtype=np.uint8)
                    
                    if tag_rygg in file:    
                        zone_mask[seg_arr > 0] = 1
                    elif tag_buk1 in file:             
                        zone_mask[seg_arr > 0] = 2
                    elif tag_buk2 in file:             
                        zone_mask[seg_arr > 0] = 3
                    elif tag_blod in file:             
                        spot_mask[seg_arr > 0] = 1
                    elif tag_melanin in file:            
                        spot_mask[seg_arr > 0] = 2
                
                task_id = os.path.basename(group[0]).split('-')[1]
                zone_output_dir = os.path.join(self.conf.get_data_dir() ,f'combined_labels/test/{key}/zone')
                spot_output_dir = os.path.join(self.conf.get_data_dir() ,f'combined_labels/test/{key}/spot')
                os.makedirs(zone_output_dir, exist_ok=True)
                os.makedirs(spot_output_dir, exist_ok=True)
                zone_output_filename = f'{ zone_output_dir}/id-{task_id}-combined-zone.npy'
                spot_output_filename = f'{spot_output_dir}/id-{task_id}-combined-spot.npy'
                np.save(zone_output_filename, zone_mask)
                np.save(spot_output_filename, spot_mask)
            print(f"Combined {len(groups)} test task groups by rater '{key}' into zone and spot masks")

    def extract_filename(self, path):
        DATA_DIR = self.conf.get_data_dir()
        fn = path.split('/')[-1]
        fn = '-'.join(fn.split('-')[1:])
        return os.path.join(DATA_DIR,'images', fn)
    
    def extract_test_filename(self, path):
        DATA_DIR = self.conf.get_data_dir()
        fn = path.split('/')[-1]
        fn = '-'.join(fn.split('-')[1:])
        return os.path.join(DATA_DIR,'images', 'test', fn)

    def extract_annotation_id(self, id, label_type):
        if label_type not in ["zone", "spot"]:
            raise ValueError("label_type must be either 'zone' or 'spot'")
        DATA_DIR = self.conf.get_data_dir()
        id_path = f'combined_labels/{label_type}/id-{id}-combined-{label_type}.npy'
        return os.path.join(DATA_DIR, id_path)
    
    def extract_test_annotation_id(self, id, label_type, rater):
        if label_type not in ["zone", "spot"]:
            raise ValueError("label_type must be either 'zone' or 'spot'")
        DATA_DIR = self.conf.get_data_dir()
        id_path = f'combined_labels/test/{rater}/{label_type}/id-{id}-combined-{label_type}.npy'
        return os.path.join(DATA_DIR, id_path)

    def file_exists(self, filepath):
        return os.path.isfile(filepath)
    
    def create_processed_data_csv(self):
        """
        Create CSV file to store information about project data. Information includes:\n
        - image_path: Path to the image file
        - zone_label_path: Path to the zone label file
        - spot_label_path: Path to the spot label file
        - seg_exists: Boolean indicating if the segmentation file exists
        - augmentation: Boolean indicating if the image is augmented
        - id: Unique identifier for the image
        - created_at: Timestamp of when the image was created
        - updated_at: Timestamp of when the image was last updated

        This informatiion is used to organize the data for easier processing and readies it for being loaded into a dataloader.\n
        """
       
        DATA_DIR = self.conf.get_data_dir()
        dir = os.path.join(DATA_DIR, 'data.csv')
        df = pd.read_csv(dir)

        df['image_path'] = df['image'].apply(self.extract_filename)
        df['zone_label_path'] = df['id'].apply(lambda x: self.extract_annotation_id(x, label_type='zone'))
        df['spot_label_path'] = df['id'].apply(lambda x: self.extract_annotation_id(x, label_type='spot'))
        df['seg_exists'] = df['spot_label_path'].apply(self.file_exists)
        df['augmentation'] = False
        df = df.drop('tag', axis=1)

        for key, file_path in self.test_csv_paths.items():
            df_test = pd.read_csv(file_path)
            df_test['image_path'] = df_test['image'].apply(self.extract_test_filename)
            df_test['zone_label_path'] = df_test['id'].apply(lambda x: self.extract_test_annotation_id(x, label_type='zone', rater=key))
            df_test['spot_label_path'] = df_test['id'].apply(lambda x: self.extract_test_annotation_id(x, label_type='spot', rater=key))
            df_test['seg_exists'] = df_test['spot_label_path'].apply(self.file_exists)
            df_test['augmentation'] = False
            df_test = df_test.drop('tag', axis=1)
            df_test.to_csv(os.path.join(DATA_DIR, f'test_processed_data_{key}.csv'), index=False)

        df.to_csv(os.path.join(DATA_DIR, 'processed_data.csv'), index=False)
        print("Created processed_data.csv for training and validation set")
        print("Created test_processed_data.csv for test set")

    def combine_csv(self):
        """
        Combines all CSV files from a directory into a single CSV file.\n
        This is used if the user has multiple CSV files from the manual annotation tool, as the project only accepts a single input csv file for processing (data.csv for training/val, test_data for testing).\n
        The csv files in data/csv_input are combined into a single data.csv file in the data directory.\n
        """
        # Get the directory containing CSV files
        csv_dir = os.path.join(self.conf.get_data_dir(), 'csv_input')
        
        # Check if directory exists
        if not os.path.exists(csv_dir):
            raise FileNotFoundError(f"CSV input directory not found: {csv_dir}")
        
        # Get list of CSV files in directory
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")
        
        # Read and combine all CSV files
        dfs = []
        for file in csv_files:
            file_path = os.path.join(csv_dir, file)
            dfs.append(pd.read_csv(file_path))
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort the dataframe by 'id' column in ascending order
        combined_df = combined_df.sort_values(by='id', ascending=True)
    
        # Save combined dataframe
        output_path = os.path.join(self.conf.get_data_dir(), 'data.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Combined {len(csv_files)} CSV files into data.csv")

    def fixTaskNumCollision(self, img_change_dir, imgs_paths, csv_path):
        """
        Util class only to be used if the user has multiple segmentation masks with the same task number due to multiple people annotating images etc.\n
        This function will rename the segmentation masks and images to avoid collisions.\n
        # Only use if you have new masks that you know have collisions with existing masks.\n
        # Using this function in other cases will cause the program to break.\n
        """
        # First, find the highest task number in imgs_path
        ref_pattern = f'task-*.png'
        ref_files = glob.glob(os.path.join(imgs_paths, ref_pattern))
        
        # Extract all task numbers and find the highest
        max_task_num = 0
        for ref_file in ref_files:
            match = re.search(r'task-(\d+)', os.path.basename(ref_file))
            if match:
                task_num = int(match.group(1))
                max_task_num = max(max_task_num, task_num)
        
        print(f"Highest task number found: {max_task_num}")

        pattern = f'task-*.png'
        matching_files = glob.glob(os.path.join(img_change_dir, pattern))

        for file_path in matching_files:
            dir_path = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            
            task_num = int(re.search(r'task-(\d+)', base_name).group(1))
            new_task_num = str(task_num + max_task_num).zfill(3)
            
            # Replace old task number with new one
            new_base_name = re.sub(r'task-\d+', f'task-{new_task_num}', base_name)
            new_name = os.path.join(dir_path, new_base_name)
            
            print(f"Renaming: {base_name} → {new_base_name}")
            os.rename(file_path, new_name)
        
        # Update CSV file
        df = pd.read_csv(csv_path)
        df['id'] = df['id'].apply(lambda x: x + max_task_num)
        df.to_csv(csv_path, index=False)
        print(f"Updated CSV file - added {max_task_num} to all IDs")

    def create_augmentations(self):
        """
        Create augmented versions of images and their corresponding segmentation masks.\n
        This function generates horizontal and vertical flips of the original images and masks, and saves them in the specified output directories (images/augmented_images),(combined_labels/augmented_labels).\n
        Running this function if the augmentations already exist will overwrite them (do nothing).\n
        """
        
        DATA_DIR = self.conf.get_data_dir()
        output_images_dir = os.path.join(DATA_DIR, 'images','augmented_images')
        output_zone_labels_dir = os.path.join(DATA_DIR, 'combined_labels','augmented_labels', 'zone')
        output_spot_labels_dir = os.path.join(DATA_DIR, 'combined_labels','augmented_labels','spot')

        dir = os.path.join(DATA_DIR, 'processed_data.csv')
        df = pd.read_csv(dir)

        #Count how many augmentations we'll create
        num_augmentations = len(df[~df['augmentation']]) * 2  # 2 for horizontal and vertical flips
        pbar = tqdm(total=num_augmentations, desc='Creating augmentations')

        
        for index, row in df.iterrows():
            image_path = row['image_path']
            zone_label_path = row['zone_label_path']
            spot_label_path = row['spot_label_path']

            if row['augmentation']:
                    continue

            img = Image.open(image_path)
            
            zone_mask = np.load(zone_label_path)
            spot_mask = np.load(spot_label_path)
            # Ensure all expected classes (0, 1, 2) are preserved during conversion
            zone_mask = torch.from_numpy(zone_mask.astype(np.int64)).long()  # Convert to tensor with long dtype for class labels
            spot_mask = torch.from_numpy(spot_mask.astype(np.int64)).long()

            # Create output directories if they don't exist
            os.makedirs(os.path.join(output_images_dir), exist_ok=True)
            os.makedirs(os.path.join(output_zone_labels_dir), exist_ok=True)
            os.makedirs(os.path.join(output_spot_labels_dir), exist_ok=True)

            # Horizontal flip
            img_h = img.transpose(Image.FLIP_LEFT_RIGHT)
            zone_mask_h = torch.flip(zone_mask, [1])
            spot_mask_h = torch.flip(spot_mask, [1])

            # Vertical flip  
            img_v = img.transpose(Image.FLIP_TOP_BOTTOM)
            zone_mask_v = torch.flip(zone_mask, [0])
            spot_mask_v = torch.flip(spot_mask, [0])

            # Save horizontal flip
            img_h.save(os.path.join(output_images_dir, 'horiz_flip_augment_' + os.path.basename(image_path)))
            np.save(os.path.join(output_zone_labels_dir, 'horiz_flip_augment_' + os.path.basename(zone_label_path)), zone_mask_h.numpy())
            np.save(os.path.join(output_spot_labels_dir, 'horiz_flip_augment_' + os.path.basename(spot_label_path)), spot_mask_h.numpy())

            # Save vertical flip
            img_v.save(os.path.join(output_images_dir, 'vert_flip_augment_' + os.path.basename(image_path)))
            np.save(os.path.join(output_zone_labels_dir, 'vert_flip_augment_' + os.path.basename(zone_label_path)), zone_mask_v.numpy())
            np.save(os.path.join(output_spot_labels_dir, 'vert_flip_augment_' + os.path.basename(spot_label_path)), spot_mask_v.numpy())

            # Create new rows for horizontal flip
            h_flip_row = row.copy()
            h_flip_row['image_path'] = os.path.join(output_images_dir, 'horiz_flip_augment_' + os.path.basename(image_path))
            h_flip_row['zone_label_path'] = os.path.join(output_zone_labels_dir, 'horiz_flip_augment_' + os.path.basename(zone_label_path))
            h_flip_row['spot_label_path'] = os.path.join(output_spot_labels_dir, 'horiz_flip_augment_' + os.path.basename(spot_label_path))
            h_flip_row['augmentation'] = True
            df = pd.concat([df, pd.DataFrame([h_flip_row])], ignore_index=True)

            # Create new rows for vertical flip
            v_flip_row = row.copy()
            v_flip_row['image_path'] = os.path.join(output_images_dir, 'vert_flip_augment_' + os.path.basename(image_path))
            v_flip_row['zone_label_path'] = os.path.join(output_zone_labels_dir, 'vert_flip_augment_' + os.path.basename(zone_label_path))
            v_flip_row['spot_label_path'] = os.path.join(output_spot_labels_dir, 'vert_flip_augment_' + os.path.basename(spot_label_path))
            v_flip_row['augmentation'] = True
            df = pd.concat([df, pd.DataFrame([v_flip_row])], ignore_index=True)

            pbar.update(2) # Update progress bar by 2 for two augmentations

        pbar.close()

        # Save updated DataFrame
        df.to_csv(os.path.join(DATA_DIR, 'processed_data.csv'), index=False)
        print(f"Added {len(df[df['augmentation']==True])} augmented images to processed_data.csv")

    



            
