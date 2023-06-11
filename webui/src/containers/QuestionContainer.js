import React, { useState } from 'react';
import { AnswerDisplay, QuestionInput } from '../components';

const QuestionContainer = () => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [isLoading, setIsLoading ] = useState(false);

    const fetchAnswer = async (question) => {
        try {
            setIsLoading(true);
            // TO ADD URL LATER
            const response = await fetch('http://localhost:5000/api/endpoint', {
                method: 'POST',
                body: JSON.stringify({ question }),
                headers: {
                'Content-Type': 'application/json',
                },
            });
            const data = await response.json();
            setAnswer(data.answer);
        } 

        catch (error) {
        console.error('Error fetching answer:', error);
        }
        finally{
            setIsLoading(false);
        }
    };

    const handleSubmit = (question) => {
        setQuestion(question);
        fetchAnswer(question);
    };

    return (
        <div>
            <QuestionInput onSubmit={handleSubmit} />
            {isLoading ? 
            (<p>Loading...</p>) : 
            (<AnswerDisplay answer={answer} />
            )}
        </div>
    );
};

export default QuestionContainer;
