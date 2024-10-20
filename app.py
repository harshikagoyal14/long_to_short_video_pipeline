import os
import tempfile
import streamlit as st
from video_process import final  # This is your `final` function

if __name__ == "__main__":
    st.title("YouTube Shorts Extractor Web App from Long Videos")

    uploaded_file = st.file_uploader("Choose a file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Process File"):
            with st.spinner("Processing..."):
                # Save the uploaded file to a temporary location
                temp_file_path = "input_file.mp4"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                try:
                    final(temp_file_path)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # Clean up temporary files
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    if os.path.exists("output_audio.wav"):
                        os.remove("output_audio.wav")
                    if os.path.exists("transcript.txt"):
                        os.remove("transcript.txt")
                    st.stop()

                # Create an output folder if it doesn't exist
                output_folder = "output"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                output_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]

                st.info("Here are the processed video clips:")

                # Display the processed video clips in a grid format
                num_columns = 3  # Number of columns in the grid
                columns = st.columns(num_columns)

                for i, output_file in enumerate(output_files):
                    with columns[i % num_columns]:
                        st.video(os.path.join(output_folder, output_file))
                        os.remove(os.path.join(output_folder, output_file))

                st.success("Processing complete!")

            # Remove temporary files after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists("output_audio.wav"):
                os.remove("output_audio.wav")
            if os.path.exists("transcript.txt"):
                os.remove("transcript.txt")
