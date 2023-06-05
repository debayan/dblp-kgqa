import './App.css';
import React, {useState} from 'react';
import { QuestionInput, AnswerDisplay } from './components';
import { Footer, Header } from './containers';

function App() {
  const [answer, setAnswer] = useState('');

  const handleAnswerReceived = (receivedAnswer) => {
    console.log('Received answer:', receivedAnswer);
    setAnswer(receivedAnswer);
  };

  return (
    <div className="App">
      <Header headingH1 = "DBLP-KGQA" headingH2="Question Answering" />
      <QuestionInput onQuestionSubmit={handleAnswerReceived}/>
      {answer && <AnswerDisplay answer = {answer} />}
      <Footer footercontent="Copyright All rights reserved 2023" />
    </div>
  );
};

export default App;
