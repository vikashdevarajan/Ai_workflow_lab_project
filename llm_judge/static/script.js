document.addEventListener('DOMContentLoaded', () => {
    const submitBtn = document.getElementById('submit-btn');
    const policyText = document.getElementById('policy-text');
    const progressContainer = document.getElementById('progress-container');
    const roundsContainer = document.getElementById('rounds-container');
    const currentStatus = document.getElementById('current-status');
    const finalResult = document.getElementById('final-result');
    const finalText = document.getElementById('final-text');
    const finalBadge = document.getElementById('final-badge');

    let currentRoundElement = null;
    let currentVotesContainer = null;

    submitBtn.addEventListener('click', async () => {
        const text = policyText.value.trim();
        if (!text) return;

        // Reset UI
        roundsContainer.innerHTML = '';
        finalResult.style.display = 'none';
        progressContainer.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.textContent = 'Refining...';
        currentStatus.textContent = 'Connecting...';
        currentStatus.style.animation = 'pulse 2s infinite';
        currentStatus.style.background = 'rgba(59, 130, 246, 0.2)';
        currentStatus.style.color = '#93c5fd';
        currentStatus.style.borderColor = 'rgba(59, 130, 246, 0.3)';
        
        currentRoundElement = null;

        try {
            const response = await fetch('/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        handleServerEvent(data);
                    }
                }
            }
        } catch (error) {
            currentStatus.textContent = 'Error: ' + error.message;
            currentStatus.style.background = 'rgba(239, 68, 68, 0.2)';
            currentStatus.style.color = '#fca5a5';
            currentStatus.style.borderColor = 'rgba(239, 68, 68, 0.3)';
            currentStatus.style.animation = 'none';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Simplify Text';
        }
    });

    function handleServerEvent(data) {
        switch (data.type) {
            case 'error':
                currentStatus.textContent = 'Error: ' + data.message;
                currentStatus.style.animation = 'none';
                currentStatus.style.background = 'rgba(239, 68, 68, 0.2)';
                currentStatus.style.color = '#fca5a5';
                break;
                
            case 'status':
                currentStatus.textContent = data.message;
                break;
                
            case 'draft':
                createRoundCard(data.round_number, data.text);
                break;
                
            case 'judge_vote':
                addJudgeVote(data);
                break;
                
            case 'round_result':
                finishRound(data);
                break;
                
            case 'final':
                showFinalResult(data);
                break;
        }
    }

    function createRoundCard(roundNum, text) {
        currentRoundElement = document.createElement('div');
        currentRoundElement.className = 'round-card';
        
        currentRoundElement.innerHTML = `
            <div class="round-header">
                <div class="round-title">Round ${roundNum}</div>
            </div>
            <div class="draft-text">${text}</div>
            <div class="judge-votes" id="votes-container-${roundNum}"></div>
            <div id="round-banner-${roundNum}"></div>
        `;
        
        roundsContainer.appendChild(currentRoundElement);
        currentVotesContainer = currentRoundElement.querySelector(`#votes-container-${roundNum}`);
        currentRoundElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    function addJudgeVote(data) {
        if (!currentVotesContainer) return;
        
        const clearHtml = `<span class="${data.clear ? 'pass' : 'fail'}">${data.clear ? 'PASS' : 'FAIL'}</span>`;
        const faithfulHtml = `<span class="${data.faithful ? 'pass' : 'fail'}">${data.faithful ? 'PASS' : 'FAIL'}</span>`;
        
        let feedbackHtml = '';
        if (data.feedback && (!data.clear || !data.faithful)) {
            feedbackHtml = `<div class="feedback-box">${data.feedback}</div>`;
        }
        
        const voteEl = document.createElement('div');
        voteEl.className = 'vote-card';
        voteEl.innerHTML = `
            <h4>Judge #${data.sample_index}</h4>
            <div class="vote-item"><span>Clear?</span> ${clearHtml}</div>
            <div class="vote-item"><span>Faithful?</span> ${faithfulHtml}</div>
            ${feedbackHtml}
        `;
        
        currentVotesContainer.appendChild(voteEl);
    }

    function finishRound(data) {
        const banner = document.getElementById(`round-banner-${data.round_number}`);
        if (!banner) return;
        
        banner.className = `round-result-banner ${data.approved ? 'approved' : 'rejected'}`;
        
        let text = data.approved ? 'Round Approved!' : 'Round Rejected (Needs Revision)';
        text += ` (Clear: ${data.clear_votes}, Faithful: ${data.faithful_votes})`;
        
        banner.textContent = text;
    }

    function showFinalResult(data) {
        currentStatus.textContent = 'Process Complete';
        currentStatus.style.animation = 'none';
        currentStatus.style.background = 'rgba(16, 185, 129, 0.2)';
        currentStatus.style.color = '#6ee7b7';
        currentStatus.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        
        finalResult.style.display = 'block';
        finalText.textContent = data.text;
        
        finalBadge.className = `badge ${data.approved ? 'approved' : 'rejected'}`;
        finalBadge.textContent = data.approved ? 'APPROVED BY JUDGES' : 'MAX ROUNDS REACHED (NOT APPROVED)';
        
        finalResult.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
});
