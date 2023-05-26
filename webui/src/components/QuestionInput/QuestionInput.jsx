import React, { useState } from 'react';
import './QuestionInput.css';

function QuestionInput() {
  const [question, setQuestion] = useState('');

  const handleInputChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    // TODO: Send the question to the backend API
    // and handle the response
    console.log('Question:', question);
  };

  return (
    <div className="QuestionForm" class="container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={handleInputChange}
          placeholder="Enter the question"
        />
        <button type="submit">Ask</button>
      </form>
    </div>
  );
}

export default QuestionInput;
