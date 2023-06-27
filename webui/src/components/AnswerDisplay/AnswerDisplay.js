import React from 'react';
import './AnswerDisplay.css';

function AnswerDisplay({answer, question}) {
  const answerNames = answer ? 
    answer.map(item => (
      <a key={item.name} href={item.url}>
        {item.name}
      </a>
    ))
    : null;

  const separatedNamesAnswer = answerNames ? answerNames.reduce((prev, curr) => [prev, ', ', curr]): null;

  return (
    <div className="answer-display-container">
      <p className="answer-title">Answer:</p>
      <div className="answer-textbox-container">
        {answer ? (
          <div className="answer-textbox">{separatedNamesAnswer} </div>
        ) : (
          <p>No answer found for the question: {question}</p>
        )}
        </div>
    </div>
  );
}

export default AnswerDisplay;