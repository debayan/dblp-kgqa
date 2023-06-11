import React from 'react'
import './footer.css';

function Footer({footercontent}){
  return (
    <footer className="App-footer">
        <p>{footercontent}</p>
      </footer>
  )
}

export default Footer;
