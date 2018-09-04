function calibrated_phase2 = phase_calibration(phasedata)
    calibrated_phase(1) = phasedata(1);
    difference = 0;
    for i = 2:30
        temp = phasedata(i) - phasedata(i-1);
        if abs(temp) > pi
            difference = difference + 1*sign(temp);
        end
        calibrated_phase(i) = phasedata(i) - difference * 2 * pi;
    end
    
    k = (calibrated_phase(30) - calibrated_phase(1)) / (30 - 1);
    b = mean(calibrated_phase);
    
    for i = 1:30
        calibrated_phase2(i) = calibrated_phase(i) - k * i - b;
    end
end