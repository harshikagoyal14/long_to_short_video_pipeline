import os
import streamlit as st
from video_process import final  # This is your `final` function
import shutil

def remove_temp_files(files):
    """Helper function to remove temporary files."""
    for file in files:
        if os.path.exists(file):
            os.remove(file)

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
                    # Run the processing function
                    final(temp_file_path)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    remove_temp_files([temp_file_path, "output_audio.wav", "transcript.txt"])
                    st.stop()

                # Create an output folder if it doesn't exist
                output_folder = "/tmp/tmpmeser_1w"  # Use your existing temporary folder here

                output_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]

                st.info("Here are the processed video clips:")

                # Display download buttons for each processed video
                for output_file in output_files:
                    file_path = os.path.join(output_folder, output_file)

                    # Display video in Streamlit
                    st.video(file_path)

                    # Create a download button for each video clip
                    with open(file_path, "rb") as video_file:
                        st.download_button(
                            label=f"Download {output_file}",
                            data=video_file,
                            file_name=output_file,
                            mime="video/mp4"
                        )

                st.success("Processing complete!")

            # Remove temporary files after processing
            remove_temp_files([temp_file_path, "output_audio.wav", "transcript.txt"])
