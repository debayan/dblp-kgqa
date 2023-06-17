import React, { useState } from 'react';
import './QuestionInput.css';

const QuestionInput = ({ onSubmit }) => {
  const [question, setQuestion] = useState('');

  const handleChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    onSubmit(question);
  };  

  return (
    <div className="question-form">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={handleChange}
          placeholder="Enter the question"
        />
        <button type="submit">Ask</button>
      </form>
    </div>
  );
};

export default QuestionInput;
