import React from 'react';
import './SparqlDisplay.css';

const SparqlDisplay = ({sparql}) => {
    return (
        <>
        <div className="sparql-display-container">
            <p className="sparql-title">SPARQL Query:</p>
            <div className="sparql-textbox-container">
                {sparql ? (
                    <div className="sparql-textbox">{sparql}</div>
                    ) : (
                        <p>No SPARQL query found for the question</p>
                )}
            </div>
        </div>
        </>
    )
}

export default SparqlDisplay;