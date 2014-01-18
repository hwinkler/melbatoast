#!/usr/bin/env ruby
# This program generates two files for input to the Gibbs sampler.
# It creates parameters for solving the case of a single reflection at
# an interface using the acoustic solution. Given two layers
# separated by an interface, and given the measured reflection amplitude
# at a range of ray parameters, determine the the layer slownesses and 
# densities of the two layers. 

# This program creates two files: a network definition, and an initial state.

require 'trollop'

opts = Trollop::options do
  opt :rhomin, "Low density", :type => :float, :required => true
  opt :rhomax, "High density",  :type => :float, :required => true
  opt :nrho, "Number of densities",  :type => :int , :default => 10 
  opt :umin, "Low slowness",  :type => :float, :required => true
  opt :umax, "High slowness", :type => :float, :required => true
  opt :nu, "Number of slownesses", :type => :int,  :default => 10
  opt :cmin, "Low reflection coefficient",  :type => :float, :required => true
  opt :cmax, "High reflection coefficient", :type => :float, :required => true
  opt :nc, "Number of reflection coefficients", :type => :int , :default => 10 

  opt :pmin, "Low ray parameter", :type => :float, :default => 0.0
  opt :pmax, "High ray parameter", :type => :float, :required => true
  opt :np, "Number of ray parameters", :type => :int, :required => true

  opt :rho1, "True density above", :type => :float, :required => true
  opt :u1, "True slowness above", :type => :float, :required => true
  opt :rho2, "True density below", :type => :float, :required => true
  opt :u2, "True slowness below", :type => :float, :required => true
 
  opt :state, "Generate a state file instead of a network"
end

Trollop::die :rhomin, "must be positive" if opts[:rhomin]  <= 0
Trollop::die :rhomax, "must be positive" if opts[:rhomax] <= 0
Trollop::die :nrho, "must be greater than one" if opts[:nrho] <= 1
Trollop::die :umin, "must be positive" if opts[:umin] <= 0
Trollop::die :umax, "must be positive" if opts[:umax] <= 0
Trollop::die :nu, "must be greater than one" if opts[:nu] <= 1
Trollop::die :cmin, "must be greater than or equal to -1" if opts[:cmin] < -1
Trollop::die :cmax, "must be less than or equal to 1" if opts[:cmax] < -1
Trollop::die :nc, "must be greater than one" if opts[:nc] <= 1

# Divide the above ranges into bins. The low of the range is the low bound of the low
# bin, and the gigh of the range is the high bound of the high bin.  That means the
# centers of the low and high bins are inset a bit from these bounds.

drho = (opts[:rhomax] - opts[:rhomin])/ opts[:nrho] 
du = (opts[:umax] - opts[:umin])/ opts[:nu] 
dc = (opts[:cmax] - opts[:cmin])/ opts[:nc] 

# ray parameter bounds don't define bins; they define the discrete values of p.
dp = (opts[:pmax] - opts[:pmin])/ (opts[:np] - 1) 


rho0 = opts[:rhomin]
u0 = opts[:pmin]
p0 = opts[:pmin]

nu = opts[:nu]
nrho = opts[:nrho]
np = opts[:np]

c0 = opts[:cmin]
nc = opts[:nc]


if opts[:state]
  u1 = opts[:u1]
  u2 = opts[:u2]
  rho1 = opts[:rho1]
  rho2 = opts[:rho2]

  puts "#{((rho1 - rho0)/drho).floor}"
  puts "#{((u1 - u0)/du).floor}"
  puts "#{((rho2 - rho0)/drho).floor}"
  puts "#{((u2 - u0)/du).floor}"

  def rfc(rayp, u1, rho1, u2, rho2)
    y1 = u1/rho1 * Complex( 1 - rayp*rayp/(u1*u1))**0.5 
    y2 = u2/rho2 * Complex( 1 - rayp*rayp/(u2*u2))**0.5
    (y1-y2)/(y1+y2)
  end

  (1..np).each do |x|
           rayp =  (x-1) * dp + p0     
           c = rfc(rayp, u1, rho1, u2, rho2)
           puts "#{((c - c0)/dc).abs.floor} f"
         end
  exit 0
end

# Make arrays for the rho and u values. Offset the values to the bin centers.
rho1a = (1..nrho).map {|x| (x-1) * drho + rho0 + drho/2} 
u1a = (1..nu).map {|x| (x-1) * du + u0 + du/2}
rho2a = (1..nrho).map {|x| (x-1) * drho + rho0 + drho/2} 
u2a = (1..nu).map {|x| (x-1) * du + u0 + du/2}

# Make an array for the p values. No need to offset here.
pa = (1..np).map {|x| (x-1) * dp + p0}

puts "RHO1 #{nrho}"
puts Array.new(nrho, 1.0/nrho).join(' ')
puts

puts "U1 #{nu}"
puts Array.new(nu, 1.0/nu).join(' ')
puts

puts "RHO2 #{nrho}"
puts Array.new(nrho, 1.0/nrho).join(' ')
puts

puts "U2 #{nu}"
puts Array.new(nu, 1.0/nu).join(' ')
puts


pa.each do |p|
    pstr = "%.6f" % p
    puts "C#{pstr.gsub('.','_')} #{nc}"

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
  end


