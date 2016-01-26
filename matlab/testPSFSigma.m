function testPSFSigma()
 w = 1:5;
 mu = 2;
 
 sigma = getPSFSigma(w,mu);
 
 for i=1:numel(sigma)
    M = normpdf(mu,mu,sigma(i));
    val1 = normpdf(mu-w(i)/2,mu,sigma(i));
    val2 = normpdf(mu+w(i)/2,mu,sigma(i));
    
    disp(abs(M-(val1+val2)))
 end

end

function sigma = getPSFSigma(w, mu)

sigma = w/(2*sqrt(2*log(2)));

end