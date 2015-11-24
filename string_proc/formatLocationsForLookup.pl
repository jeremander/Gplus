#!/usr/bin/perl -w

# formatLocationsForLookup.pl

# Shane Bergsma
# March 21, 2013

# Adapted from another script, shared with community so they can
# format their locations in a way consistent with our data.

use strict;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

while(<STDIN>) {
  my ($location) = $_;
  chomp $location;
  ########## (1) DECODE THE TWEET ############
  $location = "" if $location eq "not-given";
  $location = formatLocation($location);
  print "$location\n";
}

sub formatLocation {
  my ($location) = @_;
  return "" if $location eq "";
  # Try to clean it up:
  my @locationParts;
  $location =~ s/\p{Symbol}/ /g;
  $location =~ s/\p{Number}/ /g;
  $location =~ s/\p{Punctuation}/ /g;
  foreach my $part (split(/ /, $location)) {
	next if $part eq "";
	push(@locationParts, $part);
  }
  return "" if @locationParts == 0;
  my $loc = join(" ", @locationParts);
  $loc =~ tr/A-Z/a-z/;
  $loc =~ tr/À-Þ/à-þ/;
  return "" if (length($loc) <= 2); # But the whole thing can't be two chars or less
  return $loc;
}


