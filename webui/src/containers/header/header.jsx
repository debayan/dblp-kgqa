import React from 'react'
import './header.css';

function header({headingH1, headingH2, headingH3, headingH4, headingH5, headingH6}) {
    return (
        <header className="App-header">
            {headingH1 && <h1>{headingH1}</h1>}
            {headingH2 && <h2>{headingH2}</h2>}
            {headingH3 && <h3>{headingH3}</h3>}
            {headingH4 && <h4>{headingH4}</h4>}
            {headingH5 && <h5>{headingH5}</h5>}
            {headingH6 && <h6>{headingH6}</h6>}
        </header>
    );
}

export default header
