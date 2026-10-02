import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Smart Traffic Analytics",
    page_icon="🚗"
)

st.title("🚗 AI Smart Traffic & Vehicle Analytics")

st.write(
    "Detect vehicles from an image using YOLO."
)

uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )

    if st.button("Detect Vehicles"):

        model = YOLO("yolov8n.pt")

        image_array = np.array(image)

        results = model(
            image_array,
            verbose=False
        )

        vehicle_count = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0
        }

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name == "car":
                    vehicle_count["Car"] += 1

                elif class_name == "motorcycle":
                    vehicle_count["Motorcycle"] += 1

                elif class_name == "bus":
                    vehicle_count["Bus"] += 1

                elif class_name == "truck":
                    vehicle_count["Truck"] += 1

        st.subheader("📊 Vehicle Count")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Cars", vehicle_count["Car"])
        col2.metric("Bikes", vehicle_count["Motorcycle"])
        col3.metric("Buses", vehicle_count["Bus"])
        col4.metric("Trucks", vehicle_count["Truck"])

        result_image = results[0].plot()

        st.image(
            result_image,
            caption="Detected Vehicles",
            use_container_width=True
        )
