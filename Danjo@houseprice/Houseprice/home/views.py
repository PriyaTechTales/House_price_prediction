from django.shortcuts import render,redirect
from django.contrib import messages
from django.core.exceptions import ValidationError
import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login



# Load models with error handling
try:
    model = joblib.load('static/prediction_model')         # Bangalore model
    delhi_model = joblib.load('static/delhi_predictor')     # Delhi model
    pune_model = joblib.load('static/pune_predictor') # Pune model
except Exception as e:
    print(f"Error loading models: {e}")
    model = delhi_model = pune_model = None 

# Location mappings for one-hot encoding
location_mapping_bangalore = {
    'Whitefield': 0, 'Sarjapur Road': 1, 'Electronic City': 2, 
    'Raja Rajeshwari Nagar': 3, 'Vishwapriya Layout': 4, 'HAL 2nd Stage': 5, 
    'Thyagaraja Nagar': 6, 'Banjara Layout': 7, 'Marsur': 8
}

location_mapping_delhi = {
    'other': 0, 'J R Designers Floors, Rohini Sector 24': 1, 'Lajpat Nagar 2': 2, 
    'Lajpat Nagar 3': 3, 'The Amaryllis, Karol Bagh': 4, 'New Friends Colony': 5, 
    'Kailash Colony, Greater Kailash': 6, 'Yamuna Vihar, Shahdara': 7, 
    'Laxmi Nagar': 8, 'Patel Nagar West': 9, 'Sukhdev Vihar, Okhla': 10,
    'Saket': 11, 'Safdarjung Enclave': 12, 
    'Common Wealth Games Village, Commonwealth Games Village 2010': 13,
    'DDA Flats Sarita Vihar, Sarita Vihar, Mathura Road': 14, 'Chhattarpur': 15,
    'Mehrauli': 16, 'Mahavir Enclave Part 1': 17, 'Malviya Nagar': 18, 
    'Dilshad Colony, Dilshad Garden': 19, 'Vasundhara Enclave': 20, 
    'DLF Capital Greens, New Moti Nagar, Kirti Nagar': 21, 
    'New Moti Nagar, Kirti Nagar': 22, 'Sheikh Sarai Phase 1': 23, 
    'Hauz Khas': 24, 'Chittaranjan Park': 25
}

location_mapping_pune = {
    'Alandi Road': 1, 'Ambegaon Budruk': 2, 'Anandnagar': 3, 'Aundh': 4, 
    'Aundh Road': 5, 'Balaji Nagar': 6, 'Baner': 7, 'Baner road': 8, 
    'Bhandarkar Road': 9, 'Bhavani Peth': 10, 'Bibvewadi': 11, 'Bopodi': 12, 
    'Borivali': 13, 'Borivali East': 14, 'Borivali West': 15, 'Bhosari': 16, 
    'Budhwar Peth': 17, 'Bund Garden Road': 18, 'Camp': 19, 'Chandan Nagar': 20, 
    'Chandkheda': 21, 'Chandkheda East': 22, 'Chandkheda West': 23, 'Dapodi': 24,
    'Deccan Gymkhana': 25, 'Dehu Road': 26, 'Dhankawadi': 27, 'Dhayari Phata': 28,
    'Dhole Patil Road': 29, 'Erandwane': 30, 'Fatima Nagar': 31, 
    'Fergusson College Road': 32, 'Ganesh Peth': 33, 'Ganeshkhind': 34, 
    'Ghansopara': 35, 'Ghorpade Peth': 36, 'other': 37, 'Gokhale Nagar': 38, 
    'Gultekdi': 39, 'Hadapsar': 40, 'Hadapsar Industrial Estate': 41,
    'Hingne Khurd': 42, 'Jangali Maharaj Road': 43, 'Kalyani Nagar': 44, 
    'Karve Nagar': 45, 'Karve Road': 46, 'Kasba Peth': 47, 'Katraj': 48, 
    'Khadaki': 49, 'Kharadi': 50, 'Kondhwa': 51, 'Kondhwa Budruk': 52, 
    'Koregaon Park': 53, 'Kothrud': 54, 'Law College Road': 55, 'Laxmi Road': 56, 
    'Lulla Nagar': 57, 'Mahatma Gandhi Road': 58, 'Mangalwar peth': 59, 
    'Manik Bagh': 60, 'Market yard': 61, 'Model colony': 62, 'Mukund Nagar': 63, 
    'Mulund': 64, 'Mulund East': 65, 'Mulund West': 66, 'Nagar Road': 67, 
    'Mundhawa': 68, 'Nana Peth': 69, 'Narayan Peth': 70, 'Narayangaon': 71, 
    'Navi Peth': 72, 'Padmavati': 73, 'Parvati Darshan': 74, 'Pashan': 75, 
    'Paud Road': 76, 'Pirangut': 77, 'Prabhat Road': 78, 'Pune Railway Station': 79, 
    'Rasta Peth': 80, 'Raviwar Peth': 81, 'Sadashiv Peth': 82, 'Sahakar Nagar': 83, 
    'Salunke Vihar': 84, 'Sasson Road': 85, 'Satara Road': 86, 
    'Senapati Bapat Road': 87, 'Shaniwar Peth': 88, 'Shivaji Nagar': 89, 
    'Shukrawar Peth': 90, 'Sinhagad Road': 91, 'Somwar Peth': 92, 'Swargate': 93, 
    'Tilak Road': 94, 'Uruli Devachi': 95, 'Vadgaon Budruk': 96, 'Vadgaon Kasba': 97, 
    'Wadgaon Sheri': 98, 'Viman Nagar': 99, 'Vishrant Wadi': 100, 'Wagholi': 101, 
    'Wakadewadi': 102, 'Wanowrie': 103, 'Warje': 104, 'Yerawada': 105, 'Ghorpadi': 106
}

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[1, 1, 300], [5, 5, 5000]])) 
def about(request):
    output = None
    input_data = {}  # Initialize input_data here to avoid ReferenceError on GET request
    if request.method == 'POST':
        try:
            city = request.POST.get('city-dropdown')
            location = request.POST.get(
                'location-dropdown-bangalore' if city == 'Bangalore' else 
                'location-dropdown-delhi' if city == 'Delhi' else 
                'location-dropdown-pune' if city == 'Pune' else None
            )
            bhk = int(request.POST.get('bhk', 0))
            bathroom = int(request.POST.get('bathroom', 0))
            sqft= int(request.POST.get('sqft', 0))
            
            # Save input data for display
            input_data = {
                'city': city,
                'location': location,
                'bhk': bhk,
                'bathroom': bathroom,
                'sqft': sqft
            }

            if city == 'Delhi':
                model_to_use = delhi_model
                location_mapping = location_mapping_delhi
                input_data_array = np.zeros(4)
                location_index = location_mapping.get(location, -1)
                if location_index == -1:
                    raise ValueError("Invalid location selected")
                input_data_array[0] = bhk
                input_data_array[1] = bathroom
                input_data_array[2] = sqft
                input_data_array[3] = location_index

            elif city == 'Pune':
                model_to_use = pune_model
                location_mapping = location_mapping_pune
                input_data_array = np.zeros(4)
                location_index = location_mapping.get(location, -1)
                if location_index == -1:
                    raise ValueError("Invalid location selected")
                input_data_array[0] = bhk
                input_data_array[1] = bathroom
                input_data_array[2] = sqft
                input_data_array[3] = location_index

            else:  # Bangalore
                model_to_use = model
                location_mapping = location_mapping_bangalore
                input_data_array = np.zeros(245)
                location_index = location_mapping.get(location, -1)
                if location_index == -1:
                    raise ValueError("Invalid location selected")
                input_data_array[0] = bhk
                input_data_array[1] = bathroom
                input_data_array[2] = sqft
                if location_index >= 0:
                    input_data_array[location_index + 3] = 1

            # Predict the price
            prediction = model_to_use.predict([input_data_array])[0] if isinstance(model_to_use, BaseEstimator) else None
            
            # Adjust prediction scale based on city
            if city == 'Bangalore':
                prediction /= 1000000000  # Bangalore division
            elif city == 'Delhi':
                prediction /= 1000000  # Delhi division
            # Pune is left as is 

            output = f"Predicted Price : ₹{prediction:.2f}" if prediction else "Prediction Error"
            messages.success(request, output)
        except ValidationError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
    
    return render(request, 'about.html', {'output': output, 'input_data': input_data})


# View for home page
def index(request):
    return render(request, 'index.html')


# View for login page
def login(request):  # Renamed to avoid conflict with Django's built-in login
    return render(request, 'login.html')


# View for contact page
def contact(request):
    return render(request, 'contact.html')


# View for help page
def help(request):
    return render(request, 'help.html')


# View for signup page
def signup(request):
    return render(request, 'signup.html')