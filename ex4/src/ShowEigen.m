function ShowEigen( C )

[VC, DC] = eig(C);

lC1 = DC(1,1);
lC2 = DC(2,2);
VC1 = VC(:,1)*lC1;
VC2 = VC(:,2)*lC2;

if VC1(1) < 0 & VC1(2) < 0 
    VC1 = -1 * VC1;
end

if VC2(1) < 0 & VC2(2) < 0 
    VC2 = -1 * VC2;
end

if VC1(1) > 0 & VC1(2) < 0 
    VC1 = -1 * VC1;
end

if VC2(1) > 0 & VC2(2) < 0 
    VC2 = -1 * VC2;
end

figure;
hold on
plot( [0 VC1(1)], [0 VC1(2)], 'r', 'LineWidth', 2.0 ); 
plot( [0 VC2(1)], [0 VC2(2)], 'b', 'LineWidth', 2.0 ); 

 
legend('C1', 'C2', 'Location', 'North');

plot( 0, 0, 'ko', 'MarkerSize', 20.0 );

plot( [VC1(1)], [VC1(2)], 'ro', 'MarkerSize', 20.0 ); 
plot( [VC2(1)], [VC2(2)], 'bo', 'MarkerSize', 20.0 ); 
hold off;

axis 'equal';
return;