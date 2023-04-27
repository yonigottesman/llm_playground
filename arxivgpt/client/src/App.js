import './App.css';
import { MoonLoader } from 'react-spinners';
import axios from 'axios';
import React, { useState, useRef } from 'react';


function App() {
  const apiUrl = `${process.env.REACT_APP_BACKEND_IP}/qa`;
  const [isLoading, setIsLoading] = useState(false);
  const [answer, setAnswer] = useState("");

  const handleIndex = (event) => {
    const arxivLink = document.querySelector('input[name="arxiv_link"]').value;
    const question = document.querySelector('textarea[name="question"]').value;
    const formData = new FormData();
    formData.append('arxiv_link', arxivLink);
    formData.append('question', question);
    setIsLoading(true);
    axios.post(apiUrl, formData)
      .then(response => {
        console.log('Response from backend route:', response);
        setAnswer(response.data.answer);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error sending request to backend route:', error);
        setIsLoading(false);
      });
  };


  return (
    <div className="container">
      <div className="header">
        <h1>arxiv QA</h1>
      </div>
      <div className="rectangle">
        <input type="text" name="arxiv_link" placeholder="Enter arxiv link" />
        <textarea name="question" placeholder="Enter question" rows="5" cols="40" style={{ marginTop: '10px' }}></textarea>
        <button type="button" onClick={handleIndex}>ask</button>
        {isLoading && (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <MoonLoader color={'#FF5252'} loading={true} size={20} />
          </div>
        )}
        {answer && (
          <div style={{ marginTop: '10px' }}>
            <textarea
              value={answer}
              rows="10"
              cols="160"
              className="answer-textarea" // add the class name here
            ></textarea>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
