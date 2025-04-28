import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
import cv2
import os
import warnings
# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI
warnings.filterwarnings('ignore')

class DairyProductGrader:
    def __init__(self, data_path):
        """
        Initialize the Dairy Product Grader with the dataset path
        
        Parameters:
        -----------
        data_path : str
            Path to the milk quality dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and explore the dataset"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst 5 rows of the dataset:")
        print(self.data.head())
        print("\nData information:")
        print(self.data.info())
        print("\nStatistical summary:")
        print(self.data.describe())
        return self.data
    
    def preprocess_data(self, target_column='Grade'):
        """
        Preprocess the data for model training
        
        Parameters:
        -----------
        target_column : str
            The column name containing the grade/quality of dairy products
        """
        print("\nPreprocessing data...")
        
        # Check for missing values
        print("\nMissing values in each column:")
        print(self.data.isnull().sum())
        
        # Handle missing values if any
        if self.data.isnull().sum().sum() > 0:
            print("Filling missing values...")
            self.data = self.data.fillna(self.data.mean())
        
        # Separate features and target
        if target_column in self.data.columns:
            self.y = self.data[target_column]
            self.X = self.data.drop(columns=[target_column])
        else:
            print(f"Warning: Target column '{target_column}' not found. Using the last column as target.")
            self.y = self.data.iloc[:, -1]
            self.X = self.data.iloc[:, :-1]
        
        # Extract numerical features
        numerical_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        self.X = self.X[numerical_features]
        
        print(f"Features: {list(self.X.columns)}")
        print(f"Target: {target_column}")
        
        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        
    def analyze_features(self):
        """Analyze dataset features and their relationships"""
        print("\nPerforming feature analysis...")
        
        # Select only numeric columns for correlation
        numeric_data = self.data.select_dtypes(include=['number'])
        correlation_matrix = numeric_data.corr()
        
        # Distribution of the target variable
        plt.figure(figsize=(8, 5))
        self.y.value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribution of Dairy Product Grades')
        plt.xlabel('Grade')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('grade_distribution.png')
        plt.close()
        print("Grade distribution plot saved as 'grade_distribution.png'")
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        print("Correlation matrix saved as 'correlation_matrix.png'")
        
        # Feature distribution based on grades
        original_data = pd.DataFrame(self.X, columns=self.X.columns)
        original_data['Grade'] = self.y.values
        
        num_features = min(5, len(self.X.columns))  # Show up to 5 features
        plt.figure(figsize=(15, 8))
        for i, feature in enumerate(self.X.columns[:num_features]):
            plt.subplot(1, num_features, i+1)
            for grade in original_data['Grade'].unique():
                subset = original_data[original_data['Grade'] == grade]
                sns.kdeplot(subset[feature], label=f'Grade {grade}')
            plt.title(f'{feature} Distribution')
            plt.legend()
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.close()
        print("Feature distributions plot saved as 'feature_distributions.png'")
        
    def train_model(self, model_type='random_forest'):
        """
        Train the classification model
        
        Parameters:
        -----------
        model_type : str
            Type of model to train ('random_forest' or 'svm')
        """
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        self.model.fit(self.X_train, self.y_train)
        print(f"{model_type.capitalize()} model trained successfully!")
        
        # Feature importance (for Random Forest)
        if model_type == 'random_forest':
            feature_importance = pd.DataFrame(
                {'Feature': self.X.columns, 'Importance': self.model.feature_importances_}
            ).sort_values(by='Importance', ascending=False)
            print("\nFeature Importance:")
            print(feature_importance)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance for Dairy Product Grading')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            print("Feature importance plot saved as 'feature_importance.png'")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\nEvaluating model performance...")
        
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Grade')
        plt.ylabel('Actual Grade')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
    def analyze_sample_image(self, image_path):
        """
        Analyze a sample dairy product image
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
        """
        print(f"\nAnalyzing sample image: {image_path}")
        
        # Load and display the image
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not load image from {image_path}")
                return
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display original image
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(img_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            # Extract color features
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                plt.subplot(2, 2, i+2)
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
                plt.title(f'{color.upper()} Channel Histogram')
            
            # Calculate texture features (using grayscale image)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.subplot(2, 2, 4)
            plt.imshow(img_gray, cmap='gray')
            plt.title('Grayscale Image for Texture Analysis')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('image_analysis.png')
            plt.close()
            print("Image analysis saved as 'image_analysis.png'")
            
            # Extract and print some sample features
            avg_color = [img[:, :, i].mean() for i in range(3)]
            std_color = [img[:, :, i].std() for i in range(3)]
            
            print("\nExtracted Image Features:")
            print(f"Average RGB: {avg_color}")
            print(f"Standard Deviation RGB: {std_color}")
            
            # Advanced texture analysis
            # GLCM texture features could be added here
            # For simplicity, we're just using basic stats
            
            print("Note: In a complete system, these features would be fed to the trained model to predict quality grade")
        else:
            print(f"Error: Image file {image_path} not found")
    
    def predict_quality(self, features):
        """
        Predict the quality grade of a dairy product based on input features
        
        Parameters:
        -----------
        features : list or array
            List of features extracted from the dairy product
            
        Returns:
        --------
        quality_grade : str
            Predicted quality grade
        """
        if self.model is None:
            print("Error: Model not trained yet")
            return None
        
        # Check if the number of features matches what the model expects
        expected_features = len(self.X.columns)
        if len(features) != expected_features:
            print(f"Error: Expected {expected_features} features, but got {len(features)}")
            print(f"Expected features: {list(self.X.columns)}")
            return None
        
        # Reshape and scale the features
        features_array = np.array(features).reshape(1, -1)
        scaled_features = self.scaler.transform(features_array)
        
        # Display input features
        print("\nInput features:")
        for i, feature_name in enumerate(self.X.columns):
            print(f"{feature_name}: {features[i]}")
        
        # Predict the quality grade
        prediction = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        print(f"\nPredicted Grade: {prediction}")
        print("Prediction Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"Grade {self.model.classes_[i]}: {prob:.4f}")
        
        return prediction

def run_demo():
    """Run a demonstration of the Dairy Product Grader"""
    print("=" * 80)
    print("VISION-BASED GRADING SYSTEM FOR DAIRY PRODUCTS")
    print("=" * 80)
    
    # Initialize the grader with the dataset
    grader = DairyProductGrader('milknew.csv')
    
    # Load and explore the data
    grader.load_data()
    
    # Preprocess the data
    grader.preprocess_data(target_column='Grade')
    
    # Analyze features
    grader.analyze_features()
    
    # Train the model
    grader.train_model(model_type='random_forest')
    
    # Evaluate the model
    grader.evaluate_model()
    
    # If you have a sample image, analyze it
    # grader.analyze_sample_image('sample_milk.jpg')
    
    # Get feature information
    num_features = len(grader.X.columns)
    feature_names = list(grader.X.columns)
    print(f"\nNumber of features in model: {num_features}")
    print(f"Feature names: {feature_names}")
    
    # Create a sample with the correct number of features
    # This creates a list with the correct number of features
    sample_features = [6.8, 0.5, 3.7, 1.2, 0.8, 0.6, 0.4][:num_features]
    
    # If the sample is too short, pad it
    if len(sample_features) < num_features:
        sample_features.extend([0.5] * (num_features - len(sample_features)))
    
    grader.predict_quality(sample_features)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    run_demo()