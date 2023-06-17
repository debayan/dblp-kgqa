import React from 'react';
import './AnswerDisplay.css';

function AnswerDisplay({answer, question}) {
  const answerNames = answer ? answer.map(item => item.name).join(', ') : '';


  return (
    <div className="answer-display-container">
      <p className="answer-title">Answer:</p>
      <div className="answer-textbox-container">
        {answer ? (
          <textarea className="answer-textbox" value={answerNames} readOnly />
        ) : (
          <p>No answer found for the question: {question}</p>
        )}
        </div>
    </div>
  );
}

export default AnswerDisplay;