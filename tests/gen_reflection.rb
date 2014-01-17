drho = 0.25
du = 0.1
dp = 0.2

rho0 = 0.25
u0 = 0.1
p0 = 0.2

nu = 10
nrho = 10
np = 2

c0 = -1
dc = 0.25
nc = ((1 - c0)/dc).round +1

rho1a = (1..nrho).map {|x| (x-1) * drho + rho0} 
u1a = (1..10).map {|x| (x-1) * du + u0}
rho2a = (1..nrho).map {|x| (x-1) * drho + rho0} 
u2a = (1..10).map {|x| (x-1) * du + u0}

p = 0.4
psq = p*p

rho1a.each do |rho1| 
      u1a.each do |u1|
           y1 = u1/rho1 * Complex( 1 - psq/(u1*u1))**0.5
             rho2a.each do |rho2|
                 u2a.each do |u2|
                      #puts "#  u1=#{u1} rho1=#{rho1} u2=#{u2} rho2=#{rho2}"
 
                     y2 = u2/rho2 * Complex( 1 - psq/(u2*u2))**0.5
                     

                     if (y1+y2) == 0  && (y1+y2) == 0
                       c = 1
                     else
                       c = (y1 -y2) / (y1 + y2)
                     end
                     answer = c.abs
                     #puts "# c=#{c}  answer=#{answer}"
                     
                     slot = [ nc-1, [ 0, ((answer - c0)/dc).round].max].min 

                     #puts "# slot=#{slot}"
                     # give this slot 0.95, divide the remaining 0.05 equally
                     wslot = 0.95
                     wother = (1-wslot)/(nc-1)
                     prob = Array.new(nc, wother)
                     prob[slot] = wslot
                     prob.each {|x| print "#{x} "}
                     puts
                    end
                  end
         end
      end
