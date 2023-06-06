import React, { useState } from 'react';
import './QuestionInput.css';

const QuestionInput = ({ onQuestionSubmit }) => {
  const [question, setQuestion] = useState('');

  const handleQuestionSubmit = (event) => {
    event.preventDefault();
    onQuestionSubmit(question);
    setQuestion('');
  };  

  return (
    <div className="question-form" class="container">
      <form onQuestionSubmit={handleQuestionSubmit}>
        <input
          type="text"
          value={question}
          onChange= {(event) => setQuestion(event.target.value)}
          placeholder="Enter the question"
        />
        <button type="submit">Ask</button>
      </form>
    </div>
  );
};

export default QuestionInput;
