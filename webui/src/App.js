import logo from './logo.svg';
import './App.css';
import React from 'react';
import QuestionInput from './components/QuestionInput';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {/* <img src={logo} className="App-logo" alt="logo" /> */}
        
        {/* <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a> */}
      <h1>DBLP-KGQA</h1>
      <h2>Question Answering</h2>
      
      </header>

      <QuestionInput />
      <footer className="App-footer">
        <p>Copyright All rights reserved 2023</p>
      </footer>
    </div>
  );
}


export default App;
