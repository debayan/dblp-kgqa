import React from 'react'
import './footer.css';

function Footer({footercontent}){
  return (
    <footer className="App-footer">
        <p className="footer-text">{footercontent}</p>
      </footer>
  )
}

export default Footer;
