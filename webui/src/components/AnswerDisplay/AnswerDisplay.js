import React from 'react';
import './AnswerDisplay.css';

function AnswerDisplay({answer}) {
  return (
    <div className="answer-display" class="container">
      <h2>Answer:</h2>
      <p>{answer}</p>
    </div>
  );
}

export default AnswerDisplay;