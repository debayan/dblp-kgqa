import React, { useState } from 'react';
import './QuestionInput.css';

function QuestionInput({ onQuestionSubmit }) 
{
  const [question, setQuestion] = useState('');

  const handleInputChange = (event) => 
  {
    setQuestion(event.target.value);
  };

  const handleSubmit = async (event) => 
  {
    event.preventDefault();
    try
    {
      //ADD API URL LATER
      const response = await fetch('http://localhost:5000/api/endpoint', 
        {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ question }),
        });

      const data = await response.json();

      onQuestionSubmit(data.answer);
    } 
    
    catch (error) 
    {
      console.error('Error:', error);
    }

    // setQuestion('');
  };

  return (
    <div className="question-form" class="container">
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
};
export default QuestionInput;
