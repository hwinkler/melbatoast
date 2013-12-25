#!/usr/bin/env ruby


expected = []
actual = []
pat = /^\s*([0-9]+):\s*([0-9]+)\s*$/


class AssertionError < RuntimeError
end

def assert &block
  raise AssertionError unless yield
end

File.readlines(ARGV[0]).each do |line| 
                         m = pat.match line
                         expected.push m[2].to_f
                       end

File.readlines(ARGV[1]).each do |line| 
                         m = pat.match line
                         actual.push m[2].to_f
                       end

#normalize each 
expectedSum = expected.reduce(:+)
actualSum = actual.reduce(:+)
actual = actual.map {|x| x/actualSum}
expected = expected.map {|x| x/expectedSum}

assert { actual.length === expected.length }
actual.each_with_index do |aval, i|
        assert { (aval - expected[i]).abs < 0.05 }
end




