import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationDiceCalculator:
    """
    A class to calculate Dice scores between segmentation masks labeled by different annotators.
    
    This class handles two types of masks:
    - Zone masks: pixel values 0-3 (0=background, 1=rygg, 2=buk1, 3=buk2)
    - Spot masks: pixel values 0-2 (0=background, 1=blod, 2=melanin)
    """
    
    def __init__(self):
        self.annotator_data = {}
        self.matched_images = []
        self.zone_dice_results = {}
        self.spot_dice_results = {}
        
    def read_csv_files(self, csv_paths, annotator_names=None):
        """
        Read CSV files containing paths to segmentation masks for each annotator.
        
        Args:
            csv_paths (list): List of paths to CSV files
            annotator_names (list, optional): List of names for each annotator
        """
        if annotator_names is None:
            annotator_names = [f"Annotator_{i+1}" for i in range(len(csv_paths))]
        
        if len(csv_paths) != len(annotator_names):
            raise ValueError("Number of CSV paths must match number of annotator names")
        
        for i, (csv_path, annotator_name) in enumerate(zip(csv_paths, annotator_names)):
            try:
                df = pd.read_csv(csv_path)
                required_columns = ['image_path', 'zone_label_path', 'spot_label_path']
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    raise ValueError(f"CSV is missing required columns: {missing}")
                
                self.annotator_data[annotator_name] = df
                print(f"Loaded {len(df)} entries for {annotator_name}")
            except Exception as e:
                print(f"Error loading CSV file for {annotator_name}: {e}")
    
    def match_images(self):
        """
        Match rows across annotators that reference the same images.
        Only keep images that have been labeled by all annotators.
        """
        if len(self.annotator_data) < 2:
            raise ValueError("Need at least 2 annotators to compare")
        
        # Extract image paths for each annotator
        image_paths_by_annotator = {
            name: set(df['image_path'].apply(os.path.basename).values) 
            for name, df in self.annotator_data.items()
        }
        
        # Find common images across all annotators
        common_images = set.intersection(*image_paths_by_annotator.values())
        print(f"Found {len(common_images)} images labeled by all annotators")
        
        if not common_images:
            raise ValueError("No common images found across all annotators")
        
        # Store matched images for processing
        self.matched_images = list(common_images)
        return len(common_images)
    
    def load_mask(self, path):
        """
        Load a segmentation mask from file.
        
        Args:
            path (str): Path to the mask file
            
        Returns:
            numpy.ndarray: The loaded mask
        """
        try:
            # This assumes the masks are stored as numpy arrays
            # Modify this based on your actual file format
            mask = np.load(path)
            return mask
        except Exception as e:
            print(f"Error loading mask from {path}: {e}")
            return None
    
    def calculate_dice_score(self, mask1, mask2, class_label):
        """
        Calculate Dice score for a specific class between two masks.
        
        Args:
            mask1 (numpy.ndarray): First mask
            mask2 (numpy.ndarray): Second mask
            class_label (int): The class label to compare
            
        Returns:
            float: Dice score
        """
        if mask1 is None or mask2 is None:
            return np.nan
        
        # Create binary masks for the specific class
        binary_mask1 = (mask1 == class_label).astype(np.int8)
        binary_mask2 = (mask2 == class_label).astype(np.int8)
        
        # Calculate intersection and union
        intersection = np.sum(binary_mask1 * binary_mask2)
        sum_areas = np.sum(binary_mask1) + np.sum(binary_mask2)
        
        # Calculate Dice coefficient
        if sum_areas == 0:
            return 1.0  # Both masks agree there's no instance of this class
        else:
            return (2.0 * intersection) / sum_areas
    
    def process_all_masks(self):
        """
        Process all matched images and calculate Dice scores between all pairs of annotators.
        """
        if not self.matched_images:
            print("No matched images to process. Run match_images() first.")
            return
        
        annotator_names = list(self.annotator_data.keys())
        
        # Initialize results dictionaries for each class
        zone_classes = {1: "Rygg", 2: "Buk1", 3: "Buk2"}
        spot_classes = {1: "Blod", 2: "Melanin"}
        
        # Initialize results dictionaries
        for zone_id, zone_name in zone_classes.items():
            self.zone_dice_results[zone_name] = {}
            for i, a1 in enumerate(annotator_names):
                for a2 in annotator_names[i+1:]:
                    self.zone_dice_results[zone_name][f"{a1} vs {a2}"] = []
        
        for spot_id, spot_name in spot_classes.items():
            self.spot_dice_results[spot_name] = {}
            for i, a1 in enumerate(annotator_names):
                for a2 in annotator_names[i+1:]:
                    self.spot_dice_results[spot_name][f"{a1} vs {a2}"] = []
        
        # Process each matched image
        for img_basename in self.matched_images:
            masks_by_annotator = {}
            
            # Load masks for each annotator
            for annotator_name, df in self.annotator_data.items():
                # Find the row containing this image
                row = df[df['image_path'].apply(os.path.basename) == img_basename].iloc[0]
                
                # Load masks
                zone_mask = self.load_mask(row['zone_label_path'])
                spot_mask = self.load_mask(row['spot_label_path'])
                
                masks_by_annotator[annotator_name] = {
                    'zone': zone_mask,
                    'spot': spot_mask
                }
            
            # Calculate Dice scores between each pair of annotators
            for i, a1 in enumerate(annotator_names):
                for a2 in annotator_names[i+1:]:
                    # Zone mask comparison
                    for zone_id, zone_name in zone_classes.items():
                        dice_score = self.calculate_dice_score(
                            masks_by_annotator[a1]['zone'],
                            masks_by_annotator[a2]['zone'],
                            zone_id
                        )
                        self.zone_dice_results[zone_name][f"{a1} vs {a2}"].append(dice_score)
                    
                    # Spot mask comparison
                    for spot_id, spot_name in spot_classes.items():
                        dice_score = self.calculate_dice_score(
                            masks_by_annotator[a1]['spot'],
                            masks_by_annotator[a2]['spot'],
                            spot_id
                        )
                        self.spot_dice_results[spot_name][f"{a1} vs {a2}"].append(dice_score)
    
    def get_summary_statistics(self):
        """
        Calculate summary statistics for all Dice scores.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        stats = {
            'zone': {},
            'spot': {}
        }
        
        # Calculate statistics for zone masks
        for class_name, comparisons in self.zone_dice_results.items():
            stats['zone'][class_name] = {}
            for comparison, scores in comparisons.items():
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    stats['zone'][class_name][comparison] = {
                        'mean': np.mean(valid_scores),
                        'median': np.median(valid_scores),
                        'std': np.std(valid_scores),
                        'min': np.min(valid_scores),
                        'max': np.max(valid_scores),
                        'count': len(valid_scores)
                    }
                else:
                    stats['zone'][class_name][comparison] = {
                        'mean': np.nan, 'median': np.nan, 'std': np.nan,
                        'min': np.nan, 'max': np.nan, 'count': 0
                    }
        
        # Calculate statistics for spot masks
        for class_name, comparisons in self.spot_dice_results.items():
            stats['spot'][class_name] = {}
            for comparison, scores in comparisons.items():
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    stats['spot'][class_name][comparison] = {
                        'mean': np.mean(valid_scores),
                        'median': np.median(valid_scores),
                        'std': np.std(valid_scores),
                        'min': np.min(valid_scores),
                        'max': np.max(valid_scores),
                        'count': len(valid_scores)
                    }
                else:
                    stats['spot'][class_name][comparison] = {
                        'mean': np.nan, 'median': np.nan, 'std': np.nan,
                        'min': np.nan, 'max': np.nan, 'count': 0
                    }
        
        return stats
    
    def visualize_results(self, output_dir="."):
        """
        Create enhanced visualizations of the Dice score results including
        side-by-side mask comparisons and individual image scores.
        
        Args:
            output_dir (str): Directory to save the visualizations
        
        Returns:
            dict: Dictionary with references to poor performing images for further analysis
        """
        import cv2
        from matplotlib.colors import ListedColormap
        from matplotlib import gridspec
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Ensure we have results to visualize
        if not self.zone_dice_results or not self.spot_dice_results:
            print("No results to visualize. Run process_all_masks() first.")
            return
        
        stats = self.get_summary_statistics()
        
        # Create summary DataFrames for analysis
        zone_data = []
        for class_name, comparisons in self.zone_dice_results.items():
            for comparison, scores in comparisons.items():
                for i, score in enumerate(scores):
                    if not np.isnan(score):
                        zone_data.append({
                            'Class': class_name,
                            'Comparison': comparison,
                            'Image Index': i,
                            'Image Name': self.matched_images[i] if i < len(self.matched_images) else f"Image_{i}",
                            'Dice Score': score
                        })
        
        spot_data = []
        for class_name, comparisons in self.spot_dice_results.items():
            for comparison, scores in comparisons.items():
                for i, score in enumerate(scores):
                    if not np.isnan(score):
                        spot_data.append({
                            'Class': class_name,
                            'Comparison': comparison,
                            'Image Index': i,
                            'Image Name': self.matched_images[i] if i < len(self.matched_images) else f"Image_{i}",
                            'Dice Score': score
                        })
        
        zone_df = pd.DataFrame(zone_data)
        spot_df = pd.DataFrame(spot_data)
        
        # 1. Advanced Boxplot with Individual Points
        plt.figure(figsize=(14, 10))
        ax = sns.boxplot(x='Class', y='Dice Score', hue='Comparison', data=zone_df, palette="Set3")
        sns.stripplot(x='Class', y='Dice Score', hue='Comparison', data=zone_df, 
                      dodge=True, alpha=0.5, jitter=True, legend=False)
        plt.title('Zone Classes: Distribution of Dice Scores', fontsize=16)
        plt.ylabel('Dice Score', fontsize=14)
        plt.xlabel('Anatomical Zone', fontsize=14)
        ax.tick_params(labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path / 'zone_dice_scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(14, 10))
        ax = sns.boxplot(x='Class', y='Dice Score', hue='Comparison', data=spot_df, palette="Set2")
        sns.stripplot(x='Class', y='Dice Score', hue='Comparison', data=spot_df, 
                      dodge=True, alpha=0.5, jitter=True, legend=False)
        plt.title('Spot Classes: Distribution of Dice Scores', fontsize=16)
        plt.ylabel('Dice Score', fontsize=14)
        plt.xlabel('Spot Type', fontsize=14)
        ax.tick_params(labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path / 'spot_dice_scores_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap of mean scores with enhanced styling
        zone_classes = list(self.zone_dice_results.keys())
        spot_classes = list(self.spot_dice_results.keys())
        comparisons = [comp for comp in self.zone_dice_results[zone_classes[0]].keys()]
        
        # Zone heatmap
        zone_means = np.zeros((len(zone_classes), len(comparisons)))
        zone_stds = np.zeros((len(zone_classes), len(comparisons)))
        for i, class_name in enumerate(zone_classes):
            for j, comp in enumerate(comparisons):
                zone_means[i, j] = stats['zone'][class_name][comp]['mean']
                zone_stds[i, j] = stats['zone'][class_name][comp]['std']
        
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(zone_means, annot=True, fmt=".3f", xticklabels=comparisons, 
                    yticklabels=zone_classes, cmap="YlGnBu", vmin=0, vmax=1, 
                    annot_kws={"size": 14}, linewidths=0.5)
        
        # Add standard deviation as text in the cells
        for i in range(len(zone_classes)):
            for j in range(len(comparisons)):
                text = f"±{zone_stds[i, j]:.3f}"
                ax.text(j + 0.5, i + 0.7, text, 
                       ha="center", va="center", color="gray", fontsize=10)
                
        plt.title('Mean Dice Scores for Zone Classes (with std. dev.)', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'zone_dice_heatmap_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Spot heatmap
        spot_means = np.zeros((len(spot_classes), len(comparisons)))
        spot_stds = np.zeros((len(spot_classes), len(comparisons)))
        for i, class_name in enumerate(spot_classes):
            for j, comp in enumerate(comparisons):
                spot_means[i, j] = stats['spot'][class_name][comp]['mean']
                spot_stds[i, j] = stats['spot'][class_name][comp]['std']
        
        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(spot_means, annot=True, fmt=".3f", xticklabels=comparisons, 
                    yticklabels=spot_classes, cmap="YlGnBu", vmin=0, vmax=1,
                    annot_kws={"size": 14}, linewidths=0.5)
                    
        # Add standard deviation as text in the cells
        for i in range(len(spot_classes)):
            for j in range(len(comparisons)):
                text = f"±{spot_stds[i, j]:.3f}"
                ax.text(j + 0.5, i + 0.7, text, 
                       ha="center", va="center", color="gray", fontsize=10)
                
        plt.title('Mean Dice Scores for Spot Classes (with std. dev.)', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'spot_dice_heatmap_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Find problematic images (low dice scores)
        problematic_threshold = 0.5  # Images with dice scores below this are considered problematic
        
        # Find problematic images for zone classes
        problematic_zone_images = {}
        for class_name in zone_classes:
            problematic_zone_images[class_name] = zone_df[(zone_df['Class'] == class_name) & 
                                                       (zone_df['Dice Score'] < problematic_threshold)]
        
        # Find problematic images for spot classes
        problematic_spot_images = {}
        for class_name in spot_classes:
            problematic_spot_images[class_name] = spot_df[(spot_df['Class'] == class_name) & 
                                                       (spot_df['Dice Score'] < problematic_threshold)]
            
        # 4. Create a scatter plot for all images to visualize individual performance
        plt.figure(figsize=(15, 10))
        for i, comp in enumerate(comparisons):
            for j, cls in enumerate(zone_classes):
                subset = zone_df[(zone_df['Class'] == cls) & (zone_df['Comparison'] == comp)]
                plt.scatter(subset['Image Index'], subset['Dice Score'], 
                           label=f"{cls} - {comp}", s=50, alpha=0.7)
        
        plt.axhline(y=problematic_threshold, color='r', linestyle='-', alpha=0.5, 
                   label=f'Threshold ({problematic_threshold})')
        plt.title('Zone Classes: Dice Score by Image', fontsize=16)
        plt.xlabel('Image Index', fontsize=14)
        plt.ylabel('Dice Score', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path / 'zone_individual_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        for i, comp in enumerate(comparisons):
            for j, cls in enumerate(spot_classes):
                subset = spot_df[(spot_df['Class'] == cls) & (spot_df['Comparison'] == comp)]
                plt.scatter(subset['Image Index'], subset['Dice Score'], 
                           label=f"{cls} - {comp}", s=50, alpha=0.7)
                
        plt.axhline(y=problematic_threshold, color='r', linestyle='-', alpha=0.5, 
                   label=f'Threshold ({problematic_threshold})')
        plt.title('Spot Classes: Dice Score by Image', fontsize=16)
        plt.xlabel('Image Index', fontsize=14)
        plt.ylabel('Dice Score', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path / 'spot_individual_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Visualize mask comparisons for best and worst performing images
        # First, organize image performance for later visualization
        image_performance = {}
        for i, img_name in enumerate(self.matched_images):
            image_performance[img_name] = {'zone': {}, 'spot': {}}
            
            # Calculate average performance across all classes and comparisons for this image
            zone_scores = []
            spot_scores = []
            
            for class_name in zone_classes:
                for comp in comparisons:
                    if i < len(self.zone_dice_results[class_name][comp]):
                        score = self.zone_dice_results[class_name][comp][i]
                        if not np.isnan(score):
                            zone_scores.append(score)
                            
            for class_name in spot_classes:
                for comp in comparisons:
                    if i < len(self.spot_dice_results[class_name][comp]):
                        score = self.spot_dice_results[class_name][comp][i]
                        if not np.isnan(score):
                            spot_scores.append(score)
            
            if zone_scores:
                image_performance[img_name]['zone']['avg'] = np.mean(zone_scores)
                image_performance[img_name]['zone']['min'] = np.min(zone_scores)
                image_performance[img_name]['zone']['max'] = np.max(zone_scores)
                
            if spot_scores:
                image_performance[img_name]['spot']['avg'] = np.mean(spot_scores)
                image_performance[img_name]['spot']['min'] = np.min(spot_scores)
                image_performance[img_name]['spot']['max'] = np.max(spot_scores)
        
        # 6. Create a directory to store mask comparison images
        mask_comparison_dir = output_path / 'mask_comparisons'
        mask_comparison_dir.mkdir(exist_ok=True)
        
        # Function to visualize masks side by side
        def visualize_mask_comparison(image_name, mask_type, annotator_names, output_dir):
            """
            Create visualization of masks from different annotators side by side.
            
            Args:
                image_name: Name of the image
                mask_type: 'zone' or 'spot'
                annotator_names: List of annotator names
                output_dir: Directory to save the comparison
            """
            # Get the rows for this image from each annotator's dataframe
            rows = []
            for annotator_name in annotator_names:
                df = self.annotator_data[annotator_name]
                img_rows = df[df['image_path'].apply(os.path.basename) == image_name]
                if not img_rows.empty:
                    rows.append((annotator_name, img_rows.iloc[0]))
            
            if not rows:
                return None
                
            try:
                # Get the original image path
                original_image_path = rows[0][1]['image_path']
                
                # Try to load the original image if it exists
                try:
                    original_img = cv2.imread(original_image_path)
                    if original_img is not None:
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Could not load original image {original_image_path}: {e}")
                    original_img = None
                
                # Create figure for comparison
                n_annotators = len(rows)
                fig = plt.figure(figsize=(5 * (n_annotators + 1), 12))
                gs = gridspec.GridSpec(2, n_annotators + 1)
                
                # Create colormaps for masks
                if mask_type == 'zone':
                    colormap = ListedColormap(['black', 'red', 'green', 'blue'])
                    title_prefix = "Zone"
                    mask_path_key = 'zone_label_path'
                    class_names = {0: 'Background', 1: 'Rygg', 2: 'Buk1', 3: 'Buk2'}
                else:
                    colormap = ListedColormap(['black', 'yellow', 'magenta'])
                    title_prefix = "Spot"
                    mask_path_key = 'spot_label_path'
                    class_names = {0: 'Background', 1: 'Blod', 2: 'Melanin'}
                
                # Show original image if available
                if original_img is not None:
                    ax = plt.subplot(gs[0, 0])
                    ax.imshow(original_img)
                    ax.set_title(f"Original: {image_name}", fontsize=14)
                    ax.axis('off')
                
                # Show each annotator's mask
                for i, (annotator_name, row) in enumerate(rows):
                    # Load mask
                    mask_path = row[mask_path_key]
                    try:
                        mask = self.load_mask(mask_path)
                        
                        # Display mask
                        ax = plt.subplot(gs[0, i+1])
                        im = ax.imshow(mask, cmap=colormap, vmin=0, vmax=len(class_names)-1)
                        ax.set_title(f"{title_prefix} Mask: {annotator_name}", fontsize=14)
                        ax.axis('off')
                    except Exception as e:
                        print(f"Error displaying mask for {annotator_name}: {e}")
                
                # Show pairwise differences
                for i in range(n_annotators):
                    for j in range(i+1, n_annotators):
                        if i == 0 and j == 1:  # Only show the first pair difference in the second row
                            mask1_path = rows[i][1][mask_path_key]
                            mask2_path = rows[j][1][mask_path_key]
                            
                            try:
                                mask1 = self.load_mask(mask1_path)
                                mask2 = self.load_mask(mask2_path)
                                
                                # Create difference visualization
                                diff = np.zeros_like(mask1)
                                diff[(mask1 == mask2)] = 1  # Agreement
                                diff[(mask1 != mask2) & ((mask1 > 0) | (mask2 > 0))] = 2  # Disagreement
                                
                                ax = plt.subplot(gs[1, :])
                                diff_cmap = ListedColormap(['black', 'green', 'red'])
                                im = ax.imshow(diff, cmap=diff_cmap, vmin=0, vmax=2)
                                ax.set_title(f"Agreement Map: {rows[i][0]} vs {rows[j][0]}", fontsize=14)
                                ax.axis('off')
                                
                                # Add legend
                                from matplotlib.patches import Patch
                                legend_elements = [
                                    Patch(facecolor='black', label='Background'),
                                    Patch(facecolor='green', label='Agreement'),
                                    Patch(facecolor='red', label='Disagreement')
                                ]
                                ax.legend(handles=legend_elements, loc='upper right')
                                
                            except Exception as e:
                                print(f"Error creating difference map: {e}")
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{mask_type}_{image_name.replace('.', '_')}_comparison.png", 
                           dpi=200, bbox_inches='tight')
                plt.close(fig)
                return True
            
            except Exception as e:
                print(f"Error in mask comparison for {image_name}: {e}")
                return None
        
        # 7. Visualize best and worst performing images
        # Sort images by performance
        sorted_zone_images = sorted(
            [(img, data['zone']['avg']) for img, data in image_performance.items() if 'avg' in data['zone']],
            key=lambda x: x[1]
        )
        
        sorted_spot_images = sorted(
            [(img, data['spot']['avg']) for img, data in image_performance.items() if 'avg' in data['spot']],
            key=lambda x: x[1]
        )
        
        # Visualize 5 worst and 5 best performing images for each mask type
        n_to_visualize = min(5, len(sorted_zone_images))
        
        if sorted_zone_images:
            print(f"Visualizing {n_to_visualize} worst and best zone mask comparisons...")
            for i in range(n_to_visualize):
                # Worst
                if i < len(sorted_zone_images):
                    worst_img, score = sorted_zone_images[i]
                    visualize_mask_comparison(worst_img, 'zone', list(self.annotator_data.keys()), mask_comparison_dir)
                    print(f"  Worst zone #{i+1}: {worst_img}, Score: {score:.4f}")
                
                # Best
                if i < len(sorted_zone_images):
                    best_img, score = sorted_zone_images[-(i+1)]
                    visualize_mask_comparison(best_img, 'zone', list(self.annotator_data.keys()), mask_comparison_dir)
                    print(f"  Best zone #{i+1}: {best_img}, Score: {score:.4f}")
        
        if sorted_spot_images:
            print(f"Visualizing {n_to_visualize} worst and best spot mask comparisons...")
            for i in range(n_to_visualize):
                # Worst
                if i < len(sorted_spot_images):
                    worst_img, score = sorted_spot_images[i]
                    visualize_mask_comparison(worst_img, 'spot', list(self.annotator_data.keys()), mask_comparison_dir)
                    print(f"  Worst spot #{i+1}: {worst_img}, Score: {score:.4f}")
                
                # Best
                if i < len(sorted_spot_images):
                    best_img, score = sorted_spot_images[-(i+1)]
                    visualize_mask_comparison(best_img, 'spot', list(self.annotator_data.keys()), mask_comparison_dir)
                    print(f"  Best spot #{i+1}: {best_img}, Score: {score:.4f}")
        
        # 8. Create an interactive HTML report with JavaScript to explore images
        
        with open(output_path / 'interactive_report.html', 'w') as f:
            f.write('''<!DOCTYPE html>
        <html>
        <head>
            <title>Segmentation Agreement Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .container { max-width: 1200px; margin: 0 auto; }
                .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                tr:hover { background-color: #ddd; }
                .good { color: green; }
                .medium { color: orange; }
                .poor { color: red; }
                .image-gallery { display: flex; flex-wrap: wrap; gap: 15px; }
                .image-card { border: 1px solid #ddd; border-radius: 5px; padding: 10px; width: 280px; }
                .image-card img { width: 100%; height: auto; }
                .tabs { display: flex; margin-bottom: 10px; }
                .tab { padding: 10px 20px; cursor: pointer; background-color: #ddd; margin-right: 5px; border-radius: 5px 5px 0 0; }
                .tab.active { background-color: #4CAF50; color: white; }
                .tab-content { display: none; }
                .tab-content.active { display: block; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Segmentation Agreement Analysis</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Analysis of agreement between {len(self.annotator_data)} annotators across {len(self.matched_images)} images.</p>
                </div>
                
                <div class="tabs">
                    <div class="tab active" onclick="openTab(event, 'zone-tab')">Zone Masks</div>
                    <div class="tab" onclick="openTab(event, 'spot-tab')">Spot Masks</div>
                    <div class="tab" onclick="openTab(event, 'images-tab')">Image Comparisons</div>
                </div>
                
                <div id="zone-tab" class="tab-content active">
                    <h2>Zone Mask Agreement</h2>
                    <img src="zone_dice_scores_distribution.png" alt="Zone Dice Scores" style="max-width:100%">
                    <img src="zone_dice_heatmap_enhanced.png" alt="Zone Dice Heatmap" style="max-width:100%">
                    <img src="zone_individual_scores.png" alt="Zone Individual Scores" style="max-width:100%">
                    
                    <h3>Zone Classes Performance</h3>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Comparison</th>
                            <th>Mean Dice</th>
                            <th>Rating</th>
                        </tr>''')
                    
            # Add zone class statistics
            zone_classes = list(self.zone_dice_results.keys())
            comparisons = [comp for comp in self.zone_dice_results[zone_classes[0]].keys()]
            stats = self.get_summary_statistics()
            
            for class_name in zone_classes:
                for comp in comparisons:
                    mean_dice = stats['zone'][class_name][comp]['mean']
                    std_dice = stats['zone'][class_name][comp]['std']
                    rating = "good" if mean_dice > 0.8 else "medium" if mean_dice > 0.6 else "poor"
                    f.write(f'''
                        <tr>
                            <td>{class_name}</td>
                            <td>{comp}</td>
                            <td>{mean_dice:.4f} ± {std_dice:.4f}</td>
                            <td class="{rating}">{rating.upper()}</td>
                        </tr>''')
            
            f.write('''
                    </table>
                </div>
                
                <div id="spot-tab" class="tab-content">
                    <h2>Spot Mask Agreement</h2>
                    <img src="spot_dice_scores_distribution.png" alt="Spot Dice Scores" style="max-width:100%">
                    <img src="spot_dice_heatmap_enhanced.png" alt="Spot Dice Heatmap" style="max-width:100%">
                    <img src="spot_individual_scores.png" alt="Spot Individual Scores" style="max-width:100%">
                    
                    <h3>Spot Classes Performance</h3>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Comparison</th>
                            <th>Mean Dice</th>
                            <th>Rating</th>
                        </tr>''')
            
            # Add spot class statistics
            spot_classes = list(self.spot_dice_results.keys())
            
            for class_name in spot_classes:
                for comp in comparisons:
                    mean_dice = stats['spot'][class_name][comp]['mean']
                    std_dice = stats['spot'][class_name][comp]['std']
                    rating = "good" if mean_dice > 0.8 else "medium" if mean_dice > 0.6 else "poor"
                    f.write(f'''
                        <tr>
                            <td>{class_name}</td>
                            <td>{comp}</td>
                            <td>{mean_dice:.4f} ± {std_dice:.4f}</td>
                            <td class="{rating}">{rating.upper()}</td>
                        </tr>''')
            
            f.write('''
                    </table>
                </div>
                
                <div id="images-tab" class="tab-content">
                    <h2>Image Comparisons</h2>
                    <p>View segmentation mask comparisons between annotators:</p>
                    
                    <h3>Zone Mask Comparisons</h3>
                    <div class="image-gallery">''')
            
            # Add zone mask comparison images
            mask_comparison_dir = output_path / 'mask_comparisons'
            zone_comparisons = list(mask_comparison_dir.glob('zone_*_comparison.png'))
            
            for i, img_path in enumerate(zone_comparisons[:10]):  # Limit to first 10 images
                img_name = img_path.name.replace('zone_', '').replace('_comparison.png', '')
                img_name = img_name.replace('_', '.')  # Convert back to original file name format
                
                # Try to get the score for this image if available
                score_text = ""
                if img_name in image_performance and 'avg' in image_performance[img_name]['zone']:
                    score = image_performance[img_name]['zone']['avg']
                    score_text = f"<p>Mean Dice: {score:.4f}</p>"
                
                f.write(f'''
                        <div class="image-card">
                            <img src="mask_comparisons/{img_path.name}" alt="Zone Mask Comparison">
                            <p>Image: {img_name}</p>
                            {score_text}
                        </div>''')
            
            f.write('''
                    </div>
                    
                    <h3>Spot Mask Comparisons</h3>
                    <div class="image-gallery">''')
            
            # Add spot mask comparison images
            spot_comparisons = list(mask_comparison_dir.glob('spot_*_comparison.png'))
            
            for i, img_path in enumerate(spot_comparisons[:10]):  # Limit to first 10 images
                img_name = img_path.name.replace('spot_', '').replace('_comparison.png', '')
                img_name = img_name.replace('_', '.')  # Convert back to original file name format
                
                # Try to get the score for this image if available
                score_text = ""
                if img_name in image_performance and 'avg' in image_performance[img_name]['spot']:
                    score = image_performance[img_name]['spot']['avg']
                    score_text = f"<p>Mean Dice: {score:.4f}</p>"
                
                f.write(f'''
                        <div class="image-card">
                            <img src="mask_comparisons/{img_path.name}" alt="Spot Mask Comparison">
                            <p>Image: {img_name}</p>
                            {score_text}
                        </div>''')
            
            f.write('''
                    </div>
                </div>
            </div>

            <script>
                // Function to handle tab switching
                function openTab(evt, tabId) {
                    // Hide all tab content
                    var tabContents = document.getElementsByClassName("tab-content");
                    for (var i = 0; i < tabContents.length; i++) {
                        tabContents[i].classList.remove("active");
                    }
                    
                    // Remove active class from all tabs
                    var tabs = document.getElementsByClassName("tab");
                    for (var i = 0; i < tabs.length; i++) {
                        tabs[i].classList.remove("active");
                    }
                    
                    // Show the selected tab content and mark the clicked tab as active
                    document.getElementById(tabId).classList.add("active");
                    evt.currentTarget.classList.add("active");
                }
            </script>
        </body>
        </html>''')
    
    def save_results(self, output_dir="."):
        """
        Save all raw dice score results to CSV files.
        
        Args:
            output_dir (str): Directory to save the result files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save zone results
        for class_name, comparisons in self.zone_dice_results.items():
            df = pd.DataFrame(comparisons)
            df.to_csv(output_path / f'zone_{class_name}_dice_scores.csv', index=False)
        
        # Save spot results
        for class_name, comparisons in self.spot_dice_results.items():
            df = pd.DataFrame(comparisons)
            df.to_csv(output_path / f'spot_{class_name}_dice_scores.csv', index=False)
        
        # Save a combined summary
        stats = self.get_summary_statistics()
        
        # Create a flat representation of the statistics for easy CSV export
        rows = []
        
        for mask_type in ['zone', 'spot']:
            for class_name, comparisons in stats[mask_type].items():
                for comparison, metrics in comparisons.items():
                    row = {
                        'Mask Type': mask_type,
                        'Class': class_name,
                        'Comparison': comparison
                    }
                    row.update(metrics)
                    rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_path / 'dice_score_summary.csv', index=False)
        
        return summary_df


# Example usage
if __name__ == "__main__":
    # Initialize the calculator
    calculator = SegmentationDiceCalculator()
    
    # Example CSV paths (replace with actual paths)
    csv_paths = [
        "annotator1_labels.csv",
        "annotator2_labels.csv",
        "annotator3_labels.csv"
    ]
    
    # Example annotator names
    annotator_names = ["Person1", "Person2", "Person3"]
    
    # Read CSV files
    calculator.read_csv_files(csv_paths, annotator_names)
    
    # Match images across annotators
    num_matched = calculator.match_images()
    print(f"Processing {num_matched} matched images")
    
    # Calculate Dice scores
    calculator.process_all_masks()
    
    # Get summary statistics
    stats = calculator.get_summary_statistics()
    print("Summary Statistics:")
    for mask_type, classes in stats.items():
        print(f"\n{mask_type.upper()} MASKS:")
        for class_name, comparisons in classes.items():
            print(f"  Class: {class_name}")
            for comparison, metrics in comparisons.items():
                print(f"    {comparison}: Mean={metrics['mean']:.4f}, Median={metrics['median']:.4f}")
    
    # Create visualizations
    output_dir = "dice_score_results"
    calculator.visualize_results(output_dir)
    
    # Save all results
    calculator.save_results(output_dir)
    print(f"Results saved to {output_dir}")