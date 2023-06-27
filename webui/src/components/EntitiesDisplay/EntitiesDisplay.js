import React from 'react';
import './EntitiesDisplay.css';

const EntitiesDisplay = ({entities}) => {
    const separatedEntities = entities ? 
        entities.map((entity, index) => (
            <a key={index} href={entity}>
                {entity}
            </a>
        ))
        : null;

    return (
        <>
        <div className="entities-display-container">
            <p className="entities-title">Candidate Entities:</p>
            <div className="entities-textbox-container">
                {entities ? (
                    <div className="entities-textbox">{separatedEntities}</div>
                    ) : (
                        <p>No entity found for the question</p>
                )}
            </div>
        </div>
        </>
    )
}

export default EntitiesDisplay;