% [aeroplane]
pimap{1}('body')        = 1; 
pimap{1}('stern')       = 3;                                      
pimap{1}('lwing')        = 2;                % left wing
pimap{1}('rwing')       = 2;                % right wing
pimap{1}('tail')           = 3;                
for ii = 1:10
    pimap{1}(sprintf('engine_%d', ii)) = 4; % multiple engines
end
for ii = 1:10
    pimap{1}(sprintf('wheel_%d', ii)) = 5;  % multiple wheels
end

% [bicycle]
pimap{2}('fwheel')      = 1;                % front wheel
pimap{2}('bwheel')      = 1;                % back wheel
pimap{2}('saddle')      = 2;  
pimap{2}('handlebar')   = 3;               % handle bar
pimap{2}('chainwheel')  = 1;               % chain wheel
for ii = 1:10 
    pimap{2}(sprintf('headlight_%d', ii)) = 4;
end

% [bird]
pimap{3}('head')        = 1;
pimap{3}('leye')        = 1;                % left eye
pimap{3}('reye')        = 1;                % right eye
pimap{3}('beak')        = 1;            
pimap{3}('torso')       = 2;            
pimap{3}('neck')        = 2;            
pimap{3}('lwing')       = 3;                % left wing
pimap{3}('rwing')       = 3;               % right wing
pimap{3}('lleg')        = 4;                 % left leg
pimap{3}('lfoot')       = 4;                % left foot
pimap{3}('rleg')        = 4;                % right leg
pimap{3}('rfoot')       = 4;               % right foot
pimap{3}('tail')        = 5;
pimap{3}('head_bird') = 1;   
pimap{3}('body_4')  = 2;
pimap{3}('wing_1')  = 3;
pimap{3}('leg_4')  = 4;
pimap{3}('tail_2')        = 5;

% [boat]
% only has silhouette mask 

% [bottle]
pimap{5}('cap')          = 1;
pimap{5}('body')        = 1;


% [bus]
pimap{6}('frontside')   = 1;
pimap{6}('leftside')    = 1;
pimap{6}('rightside')   = 1;
pimap{6}('backside')    = 1;
pimap{6}('roofside')    = 1;
pimap{6}('leftmirror')  = 2;
pimap{6}('rightmirror') = 2;
pimap{6}('fliplate')    = 3;                % front license plate
pimap{6}('bliplate')    = 3;                % back license plate
for ii = 1:10
    pimap{6}(sprintf('door_%d',ii)) = 4;
end
for ii = 1:10
    pimap{6}(sprintf('wheel_%d',ii)) = 5;
end
for ii = 1:10
    pimap{6}(sprintf('headlight_%d',ii)) = 6;
end
for ii = 1:20
    pimap{6}(sprintf('window_%d',ii)) = 7;
end

pimap{6}('body_8')   = 1;
pimap{6}('mirror')  = 2;
pimap{6}('plate')    = 3;               
pimap{6}('door')    = 4;  
pimap{6}('headlight_2')    = 5; 
pimap{6}('window')    = 6; 

% [car]
keySet = keys(pimap{6});
valueSet = values(pimap{6});
pimap{7} = containers.Map(keySet, valueSet);  % car has the same set of parts with bus

% [cat]
pimap{8}('head')        = 1;
pimap{8}('leye')        = 1;                % left eye
pimap{8}('reye')        = 1;                % right eye
pimap{8}('lear')        = 1;                % left ear
pimap{8}('rear')        = 1;                % right ear
pimap{8}('nose')        = 1;
pimap{8}('torso')       = 2;   
pimap{8}('neck')        = 2;   
pimap{8}('lfleg')       = 3;                % left front leg
pimap{8}('lfpa')        = 3;               % left front paw
pimap{8}('rfleg')       = 3;               % right front leg
pimap{8}('rfpa')        = 3;               % right front paw
pimap{8}('lbleg')       = 3;               % left back leg
pimap{8}('lbpa')        = 3;               % left back paw
pimap{8}('rbleg')       = 3;               % right back leg
pimap{8}('rbpa')        = 3;               % right back paw
pimap{8}('tail')        = 4; 

% this part is defined for general parts  ;
pimap{8}('head_cat') = 1;   
pimap{8}('body_1')  = 2;  
pimap{8}('leg_1')  = 3;      

% [chair]
% only has sihouette mask 
pimap{6}('body_9')   = 1;

% [cow]
pimap{10}('head')       = 1;
pimap{10}('leye')       = 1;                % left eye
pimap{10}('reye')       = 1;                % right eye
pimap{10}('lear')       = 1;                % left ear
pimap{10}('rear')       = 1;                % right ear
pimap{10}('muzzle')     = 1;
pimap{10}('lhorn')      = 1;                % left horn
pimap{10}('rhorn')      = 1;                % right horn
pimap{10}('torso')      = 2;            
pimap{10}('neck')       = 2;
pimap{10}('lfuleg')     = 3;               % left front upper leg
pimap{10}('lflleg')     = 3;               % left front lower leg
pimap{10}('rfuleg')     = 3;               % right front upper leg
pimap{10}('rflleg')     = 3;               % right front lower leg
pimap{10}('lbuleg')     = 3;               % left back upper leg
pimap{10}('lblleg')     = 3;               % left back lower leg
pimap{10}('rbuleg')     = 3;               % right back upper leg
pimap{10}('rblleg')     = 3;               % right back lower leg
pimap{10}('tail')       = 4;               

% 
pimap{10}('head_cow') = 1;   
pimap{10}('body_2')  = 2;  
pimap{10}('leg_2')  = 3;      

% [diningtable]
% only has silhouette mask 
pimap{6}('body_10')   = 1;

% [dog]
keySet = keys(pimap{8});
valueSet = values(pimap{8});
pimap{12} = containers.Map(keySet, valueSet);         	% dog has the same set of parts with cat, 
                                            		% except for the additional
                                            		% muzzle
pimap{12}('muzzle')     = 1;

pimap{12}('head_dog') = 1;   
pimap{12}('body_1')  = 2;  
pimap{12}('leg_1')  = 3;      

% [horse]
keySet = keys(pimap{10});
valueSet = values(pimap{10});
pimap{13} = containers.Map(keySet, valueSet);        	% horse has the same set of parts with cow, 
                                                        % except it has hoof instead of horn
remove(pimap{13}, {'lhorn', 'rhorn'});
pimap{13}('lfho') = 3;
pimap{13}('rfho') = 3;
pimap{13}('lbho') = 3;
pimap{13}('rbho') = 3;
%
pimap{13}('head_horse') = 1;   
pimap{13}('body_2')  = 2;  
pimap{13}('leg_2')  = 3;      

% [motorbike]
pimap{14}('fwheel')     = 1;
pimap{14}('bwheel')     = 1;
pimap{14}('handlebar')  = 3;
pimap{14}('saddle')     = 2;
for ii = 1:10 
    pimap{14}(sprintf('headlight_%d', ii)) = 4;
end 

pimap{14}('wheel')  = 1;
pimap{14}('saddle')  = 2;
pimap{14}('handlebar')  = 3;
pimap{14}('headlight_1')  = 4;

% [person]
pimap{15}('head')       = 1;
pimap{15}('leye')       = 1;                   % left eye
pimap{15}('reye')       = 1;                   % right eye
pimap{15}('lear')       = 1;                    % left ear
pimap{15}('rear')       = 1;                    % right ear
pimap{15}('lebrow')     = 1;                  % left eyebrow    
pimap{15}('rebrow')     = 1;                    % right eyebrow
pimap{15}('nose')       = 1;                    
pimap{15}('mouth')      = 1;                    
pimap{15}('hair')       = 1;                   

pimap{15}('torso')      = 2;                   
pimap{15}('neck')       = 2;           
pimap{15}('llarm')      = 3;                   % left lower arm
pimap{15}('luarm')      = 3;                   % left upper arm
pimap{15}('lhand')      = 3;                   % left hand
pimap{15}('rlarm')      = 3;                   % right lower arm
pimap{15}('ruarm')      = 3;                   % right upper arm
pimap{15}('rhand')      = 3;                   % right hand

pimap{15}('llleg')      = 4;               	% left lower leg
pimap{15}('luleg')      = 4;               	% left upper leg
pimap{15}('lfoot')      = 4;               	% left foot
pimap{15}('rlleg')      = 4;               	% right lower leg
pimap{15}('ruleg')      = 4;               	% right upper leg
pimap{15}('rfoot')      = 4;               	% right foot


% [pottedplant]
pimap{16}('pot')        = 1;
pimap{16}('plant')      = 2;

% [sheep]
keySet = keys(pimap{10});
valueSet = values(pimap{10});
pimap{17} = containers.Map(keySet, valueSet);        % sheep has the same set of parts with cow

% [sofa]
% only has sihouette mask 

% [train]
pimap{19}('head')       = 1;
pimap{19}('hfrontside') = 1;                	% head front side                
pimap{19}('hleftside')  = 1;                	% head left side
pimap{19}('hrightside') = 1;                	% head right side
pimap{19}('hbackside')  = 1;                 	% head back side
pimap{19}('hroofside')  = 1;                	% head roof side

for ii = 1:10
    pimap{19}(sprintf('headlight_%d',ii)) = 2;
end

for ii = 1:10
    pimap{19}(sprintf('coach_%d',ii)) = 1; % I think coach belongs to instance s
end

for ii = 1:10
    pimap{19}(sprintf('cfrontside_%d', ii)) = 1;   % coach front side
end

for ii = 1:10
    pimap{19}(sprintf('cleftside_%d', ii)) = 1;   % coach left side
end

for ii = 1:10
    pimap{19}(sprintf('crightside_%d', ii)) = 1;  % coach right side
end

for ii = 1:10
    pimap{19}(sprintf('cbackside_%d', ii)) = 1;   % coach back side
end

for ii = 1:10
    pimap{19}(sprintf('croofside_%d', ii)) = 1;   % coach roof side
end
pimap{19}('body_8') = 1;
pimap{19}('headlight_2') = 1; 

% [tvmonitor]
pimap{20}('screen')     = 1;
