import streamlit as st
import tensorflow as tf
import numpy as np 

# Define Wikipedia links, descriptions, and fertilizer recommendations for each disease
DISEASE_INFO = {
    'Potato___Early_blight': {
        'description': "Early blight is a fungal disease caused by the fungus Alternaria solani. It typically manifests as circular brown spots with concentric rings on the lower leaves of the plant. Early blight can lead to significant yield losses if not managed properly. It thrives in warm, humid conditions and can spread rapidly during periods of high moisture. The disease is often observed during the mid to late stages of crop development.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Early_blight_of_potato",
        'fertilizer_link': "https://krishisevakendra.in/products/metalaxyl-8-mancozeb-64-wp-meta-manco-fungicide?variant=45290989191464&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&gad_source=4&gclid=Cj0KCQjw6uWyBhD1ARIsAIMcADobfS3P0jIWKm9oTHh1eUXDihAE4y0X1Bhx2K6snYLhBbl75sFhmhMaAnzfEALw_wcB",
        'natural_treatment': "To manage early blight naturally, prune affected leaves and avoid overhead watering to reduce moisture on the leaves. Use organic sprays like neem oil, garlic extract, or a mixture of baking soda and water. These organic treatments help suppress fungal growth and protect the plant from further infection. Additionally, promoting good air circulation around the plants by proper spacing and removing weeds can help prevent the spread of the disease."
    },
    'Potato___healthy': {
        'description': "No disease detected. The plant appears to be healthy.",
        'wikipedia_link': None,
        'fertilizer_link': None,
        'natural_treatment': None
    },
    'Potato___Late_blight': {
        'description': "Late blight, caused by the oomycete pathogen Phytophthora infestans, is one of the most destructive diseases affecting potatoes. Symptoms include dark, water-soaked lesions on the leaves, stems, and fruits. Late blight thrives in cool, wet conditions and can spread rapidly under favorable weather. The disease can cause extensive damage to potato crops if left untreated.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Late_blight",
        'fertilizer_link': "https://www.google.com/shopping/product/1?q=potato+late+blight++disease+fertilizer&prds=epd:17278135005082470878,eto:17278135005082470878_0,pid:17278135005082470878&sa=X&ved=0ahUKEwjMurviq7iGAxUtxjgGHaG0BFEQ9pwGCAg",
        'natural_treatment': "For late blight, use resistant plant varieties and remove infected plant parts promptly to prevent the spread of the disease. Apply copper-based fungicides as a preventive measure, especially during periods of high humidity. Additionally, avoid overhead watering and ensure proper drainage to minimize moisture on the leaves. Organic treatments such as compost tea or milk spray can also help suppress the disease and boost plant immunity."
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': "Gray leaf spot, caused by the fungus Cercospora zeae-maydis, appears as small, rectangular lesions with gray centers and dark-brown borders on the leaves. This disease primarily affects maize crops and can lead to significant yield losses if not managed effectively. Gray leaf spot thrives in warm, humid conditions and spreads rapidly through splashing water and wind-dispersed spores.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Cercospora_leaf_spot",
        'fertilizer_link': "https://www.google.com/shopping/product/1?q=maize+Gray_leaf_spot+fertilizer&prds=epd:10302831533423480094,eto:10302831533423480094_0,pid:10302831533423480094&sa=X&ved=0ahUKEwjXp_KtrLiGAxXbbGwGHbcNAp0Q9pwGCAU",
        'natural_treatment': "To control gray leaf spot naturally, adopt integrated pest management practices such as crop rotation and intercropping with legumes to disrupt disease cycles. Remove crop debris after harvest to reduce overwintering fungal spores. Additionally, maintain optimal plant nutrition and avoid excessive nitrogen fertilization, as it can increase susceptibility to the disease. Apply organic treatments such as compost tea or botanical extracts like neem oil to suppress fungal growth and strengthen plant immunity."
    },
    'Corn_(maize)___Common_rust_': {
        'description': "Common rust, caused by the fungus Puccinia sorghi, is a prevalent disease affecting maize crops worldwide. It appears as small, circular, reddish-brown pustules on the leaves, which can coalesce under favorable conditions. Common rust can reduce photosynthetic efficiency and weaken plants, leading to yield losses if left unmanaged.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Common_rust",
        'fertilizer_link': "https://samalagrotech.in/product/3367046/Hyzinc-fertilizer-1kg?utm_source=GMC",
        'natural_treatment': "Manage common rust by planting resistant maize varieties and practicing good cultural practices such as proper spacing and weed management. Remove infected leaves and destroy crop residues to reduce overwintering spores. Apply organic fungicides containing copper or sulfur to protect plants from infection. Additionally, promote plant vigor through balanced nutrition and soil amendments to enhance disease resistance."
    },
    'Corn_(maize)___healthy': {
        'description': "No disease detected. The maize plant appears to be healthy.",
        'wikipedia_link': None,
        'fertilizer_link': None,
        'natural_treatment': None
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': "Northern leaf blight is caused by the fungus Exserohilum turcicum and appears as large, elliptical lesions with tan centers and dark-brown borders on the leaves. The disease can significantly reduce yields if not managed properly. It thrives in warm, humid conditions and spreads rapidly through wind-dispersed spores. Northern leaf blight is commonly observed during the mid to late stages of crop development.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Northern_leaf_blight_(maize)",
        'fertilizer_link': "https://www.amazon.in/Ebee-Leaf-Compost-Patte-15KG/dp/B0BPQQDVC1?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A2AL6IVND0I91F",
        'natural_treatment': "To manage northern leaf blight, remove and destroy infected plant debris, use resistant maize varieties, and apply neem oil as an organic fungicide. Implement cultural practices such as proper spacing and weed management to reduce disease pressure. Additionally, avoid overhead watering and promote good air circulation around plants to minimize moisture on the leaves. Organic treatments such as compost tea or microbial inoculants can also help boost plant immunity and suppress fungal growth."
    },
    'Grape___Black_rot': {
        'description': "Black rot, caused by the fungus Guignardia bidwellii, appears as circular lesions on the leaves, which later turn brown and develop black fungal fruiting bodies. This disease is prevalent in warm, humid regions and can cause significant economic losses to grape growers if not managed effectively. Black rot can affect all parts of the grapevine, including leaves, stems, and fruit clusters.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Black_rot_(grape_disease)",
        'fertilizer_link': "https://www.flipkart.com/home-garden-grapes-food-essential-organic-fertilizer-plant/p/itma431e71b25c4a?pid=SMNGGMQQ4BKMFEBT&lid=LSTSMNGGMQQ4BKMFEBTYTBHYZ&marketplace=FLIPKART&cmpid=content_soil-manure_8965229628_gmc",
        'natural_treatment': "To control black rot, remove infected plant parts, ensure good air circulation, and apply sulfur-based organic fungicides. Prune the grapevines to improve air circulation and reduce humidity around the plants. Additionally, maintain good sanitation by removing and destroying fallen leaves and fruit to reduce sources of infection. Organic treatments such as neem oil, baking soda sprays, or compost tea can help suppress the fungus and protect the grapevines."
    },
    'Grape___Esca_(Black_Measles)': {
        'description': "Esca, or black measles, is caused by several different fungi and manifests as leaf discoloration, shoot dieback, and black streaks in the wood. This disease can severely impact grapevine health and productivity. Esca is often associated with older vineyards and can persist in the soil for many years.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Esca_(grape_disease)",
        'fertilizer_link': "https://www.flipkart.com/sansar-green-grapes-fertilizer/p/itm90f06136b50c7?pid=SMNFNTDKCBPZGSMM&lid=LSTSMNFNTDKCBPZGSMMNFEUTT&marketplace=FLIPKART&cmpid=content_soil-manure_8965229628_gmc",
        'natural_treatment': "For esca, prune infected plant parts and promote good air circulation. Apply organic treatments like compost teas to strengthen plant health and improve soil microbiology. Ensure proper irrigation practices to avoid water stress, which can exacerbate the disease. Additionally, mulch around the grapevines to maintain soil moisture and temperature, and apply organic fertilizers to improve overall plant vigor."
    },
    'Grape___healthy':{
        'description': "Leaf blight, caused by the fungus Isariopsis spp., appears as small, circular lesions on the leaves that gradually enlarge and turn brown. This disease can lead to premature leaf drop and reduce the photosynthetic capacity of the grapevines, impacting yield and fruit quality.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Grape_diseases",
        'fertilizer_link': "https://leaffyco.com/product/organic-leaf-compost-fertilizer/?attribute_weight=500G",
        'natural_treatment': "To manage leaf blight, remove infected leaves, ensure good air circulation, and apply neem oil or garlic extract as organic fungicides. Implement cultural practices such as proper spacing and pruning to improve air flow and reduce humidity around the plants. Additionally, maintain good sanitation by removing fallen leaves and debris. Organic treatments such as baking soda sprays or compost tea can help suppress the fungal pathogen and protect the grapevines."
    },
    
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': "No disease detected. The grapevine appears to be healthy.",
        'wikipedia_link': None,
        'fertilizer_link': None,
        'natural_treatment': None 
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': "Bacterial spot, caused by the bacterium Xanthomonas campestris pv. vesicatoria, appears as small, water-soaked lesions on the leaves, which later turn dark brown or black. This disease can spread rapidly under warm, humid conditions and can cause significant yield losses if not managed properly. Bacterial spot can affect both leaves and fruit, leading to reduced marketability.",
        'wikipedia_link': "https://en.wikipedia.org/wiki/Bacterial_spot_of_pepper_and_tomato",
        'fertilizer_link': "https://www.google.com/shopping/product/3008973371124237638?q=pepper+leaf+spot+fertilizer&prds=eto:14407869479571955412_0;7753491026450609760_0,pid:7464095628889804525&sa=X&ved=0ahUKEwiZtP2RrriGAxW_xDgGHUznC5oQ9pwGCAw",
        'natural_treatment': "Control bacterial spot by removing infected plant debris, ensuring good air circulation, and applying copper-based organic fungicides. Implement crop rotation and avoid planting peppers in the same location year after year. Additionally, use disease-free seeds and transplants to prevent introduction of the pathogen. Organic treatments such as garlic or neem oil sprays can help reduce bacterial populations. Maintain good sanitation by cleaning tools and equipment to prevent the spread of bacteria."
    },
    'Pepper,_bell___healthy': {
        'description': "No disease detected. The bell pepper plant appears to be healthy.",
        'wikipedia_link': None,
        'fertilizer_link': None,
        'natural_treatment': None
    }
}

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(f"./2227kooo.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256,256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return predictions


# Streamlit App
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the App Mode", ["Home", "About App", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# About Page
elif app_mode == "About App":
    st.title("About the Plant Disease Recognition App")
    st.markdown("""
    This application is designed to help farmers and gardeners identify plant diseases using deep learning. 
    Upload a clear image of a plant's leaves, and the model will predict the disease affecting the plant.
    
    ### Features:
    - Predict plant diseases from images.
    - Provides information about the predicted disease.
    - Recommends treatments and fertilizers.
    - Supports Grape, Potato, Pepper, and Maize plants.
    
    ### How to Use:
    1. Navigate to the **Disease Recognition** section.
    2. Upload an image of the plant's leaves.
    3. Click on the **Predict** button to see the results.
    4. Read about the disease and follow the recommended treatment.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    st.markdown("""
    <p style='font-size: 14px;'>Upload a clear image of the plant's leaves. Ensure the image is focused and well-lit for better accuracy. Supported plants include Grape, Potato, Pepper, and Maize.</p>
    <p style='font-size: 14px;'>Following these guidelines will help improve the accuracy of the prediction.</p>
    """, unsafe_allow_html=True)



    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.spinner()
        
        st.write("Our Prediction and Information")
        predictions = model_prediction(test_image)
        result_index = np.argmax(predictions)
        class_name = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        predicted_disease = class_name[result_index]
        accuracy = np.max(predictions) * 99
        st.success("Model is Predicting it's a {} with {:.2f}% accuracy".format(predicted_disease, accuracy))
    
        # Display disease description, Wikipedia link, and fertilizer recommendation if available
        disease_info = DISEASE_INFO.get(predicted_disease, {})
        description = disease_info.get('description', "No description available.")
        wikipedia_link = disease_info.get('wikipedia_link')
        fertilizer_link = disease_info.get('fertilizer_link')
        natural_treatment = disease_info.get('natural_treatment', None)
        
        if wikipedia_link:
            st.markdown(f"**Description**: {description}")
            st.write(f"Learn more about {predicted_disease} [here]({wikipedia_link}).")
            if fertilizer_link:
                st.write(f"To treat {predicted_disease}, it is recommended to use a suitable fertilizer. You can purchase it [here]({fertilizer_link}).")
            if natural_treatment:
                st.markdown(f"### Natural Treatment")
                st.write(natural_treatment)
        else:
            st.write(description)
