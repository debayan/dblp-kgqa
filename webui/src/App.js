import './App.css';
import React from 'react';
import {Footer, Header } from './components';
import { QuestionContainer } from './containers';

function App() {
  // const [answer, setAnswer] = useState('')

  // const handleAnswerReceived = (receivedAnswer) => {
  //   console.log('Received answer:', receivedAnswer);
  //   setAnswer(receivedAnswer);
  // };

  return (
    <div className="App">
      < Header headingH1 = "DBLP-KGQA" headingH2="Question Answering" />
      <QuestionContainer />
      <Footer footercontent="Copyright All rights reserved 2023" />   
    </div>
  );
};

export default App;
