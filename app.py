import os
import streamlit as st
from video_process import final

def remove_temp_files(files):
    """Helper function to remove temporary files."""
    for file in files:
        if os.path.exists(file):
            os.remove(file)

def main():
    st.title("YouTube Shorts Extractor Web App from Long Videos")

    uploaded_file = st.file_uploader("Choose a file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Process File"):
            with st.spinner("Processing..."):
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(tempfile.gettempdir(), "input_file.mp4")
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                try:
                    # Access API keys from Streamlit secrets
                    google_api_key = st.secrets["google_api"]["api_key"]
                    assembly_api_key = st.secrets["assembly_ai"]["api_key"]

                    # Run the processing function
                    output_files = final(temp_file_path, google_api_key, assembly_api_key)

                    if output_files:
                        st.success("Processing complete! Downloading files...")
                        for output_file in output_files:
                            with open(output_file, "rb") as file:
                                st.download_button(
                                    label=f"Download {os.path.basename(output_file)}",
                                    data=file,
                                    file_name=os.path.basename(output_file),
                                    mime="video/mp4"
                                )
                    else:
                        st.warning("No output files found.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    remove_temp_files([temp_file_path])

if __name__ == "__main__":
    main()
