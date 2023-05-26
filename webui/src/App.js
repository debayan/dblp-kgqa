
import './App.css';
import React from 'react';
import { QuestionInput } from './components';
import { Footer, Header } from './containers';

function App() {
  return (
    <div className="App">
      <Header headingH1 = "DBLP-KGQA" headingH2="Question Answering" />
      <QuestionInput />
      <Footer footercontent="Copyright All rights reserved 2023" />
    </div>
  );
}


export default App;
