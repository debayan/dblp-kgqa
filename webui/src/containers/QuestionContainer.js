import React, { useState } from 'react';
import axios from 'axios';
import { AnswerDisplay, QuestionInput } from '../components';

const QuestionContainer = () => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [entities, setEntities] = useState([]);
    const [sparql, setSparql] = useState('');
    const [isLoading, setIsLoading ] = useState(false);
    const [isAnswerVisible, setIsAnswerVisible] = useState(false);

    const fetchAnswer = async (question) => {
        setIsLoading(true);
        try {
            // TO ADD URL LATER
            const response = await axios.post('http://127.0.0.1:5000/api/endpoint', 
                {
                    question: question
                },
                {
                    headers: {
                    'Content-Type': 'application/json',
                    },
                }
            );
            const { answer, entities, sparql } = response.data;
            
            setAnswer(answer);
            setEntities(entities);
            setSparql(sparql);
            setIsAnswerVisible(true);
        } 

        catch (error) {
            console.error('Error fetching answer:', error);
        }
        finally{
            setIsLoading(false);
        }
    };

    const handleSubmit = async (question) => {
        setQuestion(question);
        setIsAnswerVisible(false);
        fetchAnswer(question);
    };

    return (
        <div>
            <QuestionInput onSubmit= {handleSubmit} />
            {isLoading ? (
            <p>Loading...</p>
            ) : isAnswerVisible ? 
            (<AnswerDisplay answer = {answer} />
            ) : null}
        </div>
    );
};

export default QuestionContainer;
