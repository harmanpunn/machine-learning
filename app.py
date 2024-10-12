import streamlit as st
import json

# Load the JSON file
def load_transcribe_json(file):
    data = json.load(file)
    return data

# Extract the transcript and speaker information
def parse_transcribe_data(data):
    transcript_text = data['results']['transcripts'][0]['transcript']
    speaker_segments = data['results'].get('speaker_labels', {}).get('segments', [])

    # Prepare a list to hold the conversation
    conversation = []

    for segment in speaker_segments:
        speaker = segment['speaker_label']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # Get the text associated with this speaker segment
        text = []
        for item in segment['items']:
            if item.get('alternatives'):
                text.append(item['alternatives'][0]['content'])
        
        # Combine the texts
        segment_text = ' '.join(text)
        
        # Append to the conversation list
        conversation.append({"speaker": speaker, "start_time": start_time, "end_time": end_time, "text": segment_text})

    return transcript_text, conversation

# Streamlit App
def main():
    st.title("Amazon Transcribe Conversation Viewer")
    
    uploaded_file = st.file_uploader("Upload Amazon Transcribe JSON file", type=["json"])
    
    if uploaded_file:
        # Load the JSON data
        data = load_transcribe_json(uploaded_file)

        # Parse the data
        transcript_text, conversation = parse_transcribe_data(data)

        # Display the transcript
        st.header("Full Transcript")
        st.write(transcript_text)

        # Display the conversation with speaker labels
        st.header("Conversation by Speaker")
        for entry in conversation:
            st.subheader(f"{entry['speaker']} (Start: {entry['start_time']} - End: {entry['end_time']})")
            st.write(entry['text'])

# Run the Streamlit app
if __name__ == "__main__":
    main()
