document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const recordButton = document.getElementById('recordButton');
  const statusDisplay = document.getElementById('status');
  const incidentForm = document.getElementById('incidentForm');
  const formFields = {
    location: document.getElementById('location'),
    incident_type: document.getElementById('incident_type'),
    severity: document.getElementById('severity'),
    description: document.getElementById('description'),
    actions_taken: document.getElementById('actions_taken')
  };

  // --- State Variables ---
  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;
  let currentIncidentData = null;  // Stores incomplete data for the merge pass

  // --- Main Record Button Logic ---
  recordButton.addEventListener('click', () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });

  // --- Recording Functions ---
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, {type: 'audio/webm'});
        audioChunks = [];
        sendAudioToServer(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      updateUIForRecording(true);
    } catch (error) {
      statusDisplay.textContent = 'Error: Could not access microphone.';
      console.error('Mic error:', error);
    }
  }

  function stopRecording() {
    if (mediaRecorder) {
      mediaRecorder.stop();
      updateUIForRecording(false);
      statusDisplay.textContent = 'Status: Processing... Please wait.';
    }
  }

  // --- UI Update Function ---
  function updateUIForRecording(isRec) {
    isRecording = isRec;
    recordButton.textContent =
        isRec ? '🛑 Stop Recording' : '🎤 Start Recording';
    recordButton.classList.toggle('recording', isRec);
    statusDisplay.textContent =
        isRec ? 'Status: Recording...' : 'Status: Ready';
  }

  // --- Server Communication ---
  async function sendAudioToServer(audioBlob) {
    const formData = new FormData();
    formData.append('audio_data', audioBlob, 'incident.webm');

    // If we have partial data, send it as context for the merge pass
    if (currentIncidentData) {
      formData.append('context', JSON.stringify(currentIncidentData));
    }

    try {
      const response =
          await fetch('/process_incident', {method: 'POST', body: formData});
      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

      const data = await response.json();
      handleServerResponse(data.parsed_incident);

    } catch (error) {
      statusDisplay.textContent = 'Error: Failed to process audio.';
      console.error('Server error:', error);
    }
  }

  // --- Handle LLM Response and TTS ---
  function handleServerResponse(parsedData) {
    currentIncidentData = parsedData;  // Store the latest data state
    updateFormFields(parsedData);

    const missingFields =
        Object.keys(parsedData).filter(key => parsedData[key] === null);

    if (missingFields.length > 0) {
      const promptText = `Please provide the following missing information: ${
          missingFields.join(', ')}.`;
      statusDisplay.textContent =
          `Awaiting info for: ${missingFields.join(', ')}`;
      speak(promptText);  // Use Text-to-Speech
    } else {
      statusDisplay.textContent = 'Status: All information captured!';
      currentIncidentData = null;  // Reset for next incident
    }
  }

  // --- Helper Functions ---
  function updateFormFields(data) {
    for (const key in formFields) {
      if (data[key] && formFields[key]) {
        formFields[key].value = data[key];
      }
    }
  }

  async function speak(text) {
    try {
      const response = await fetch('/synthesize_speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: text}),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch speech audio from server.');
      }

      // Get the audio data as a blob
      const audioBlob = await response.blob();
      // Create a URL for the blob
      const audioUrl = URL.createObjectURL(audioBlob);
      // Create a new Audio object and play it
      const audio = new Audio(audioUrl);
      audio.play();

    } catch (error) {
      console.error('Error with Text-to-Speech:', error);
      // Fallback to the browser's basic voice if the API fails
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(
            'Error generating high-quality voice. ' + text);
        speechSynthesis.speak(utterance);
      }
    }
  }

  // --- Manual Form Submission ---
  incidentForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData(incidentForm);
    const report = Object.fromEntries(formData.entries());
    console.log('Manual Form Submitted:', report);
    alert('Incident report submitted successfully!');
    incidentForm.reset();
    currentIncidentData = null;  // Reset state after submission
    statusDisplay.textContent = 'Status: Ready';
  });
});