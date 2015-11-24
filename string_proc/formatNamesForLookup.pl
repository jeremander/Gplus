#!/usr/bin/perl -w

# formatNamesForLookup.pl

# Shane Bergsma
# November 19, 2012

use strict;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";

while (<STDIN>) {
  my $name = $_;
  chomp $name;

  my @nameParts;
	# No numbers, punctuation, or special symbols in your name
  $name =~ s/\p{Symbol}/ /g;
  $name =~ s/\p{Number}/ /g;
  $name =~ s/\p{Punctuation}/ /g;
  # Now switch to lower case:
  $name =~ tr/A-Z/a-z/;
  $name =~ tr/À-Þ/à-þ/;
  # Here are some rules for first names with prefixes:
  $name =~ s/\b(ms)[.]?\b//g;
  $name =~ s/\b(miss)[.]?\b//g;
  $name =~ s/\b(mrs)[.]?\b//g;
  $name =~ s/\b(sr)[.]?\b//g;
  $name =~ s/\b(sra)[.]?\b//g;
  $name =~ s/\b(srta)[.]?\b//g;
  $name =~ s/\b(mr)[.]?\b//g;
  $name =~ s/\b(sir)[.]?\b//g;
  $name =~ s/\b(dr)[.]?\b//g;
  $name =~ s/\b(rev)[.]?\b//g;

  #d’ D’ de, De, Del, De la, Di, Du, El, Fits, La, Le, M, Mac, Mc, O’, Saint, St. Van, Van de, Van der, Von, and Von der.
  #M, Mac, Mc, Saint, St. Van, Van de, Van der, Von, and Von der.
  # Here are some rules for last names with prefixes:
  $name =~ s/ d[`']/ d~/g;
  $name =~ s/ de / de~/g;
  $name =~ s/ del / del~/g;
  $name =~ s/ de la / de~la~/g;
  $name =~ s/ di / di~/g;
  $name =~ s/ du / du~/g;
  $name =~ s/ el / el~/g;
  $name =~ s/ fits / fits~/g;
  $name =~ s/ la / la~/g;
  $name =~ s/ le / le~/g;
  $name =~ s/ m / m~/g;
  $name =~ s/ mac / mac~/g;
  $name =~ s/ mc / mc~/g;
  $name =~ s/ saint / saint~/g;
  $name =~ s/ st\. / st.~/g;
  $name =~ s/ van / van~/g;
  $name =~ s/ van de / van~de~/g;
  $name =~ s/ van der / van~der~/g;
  $name =~ s/ von / von~/g;
  $name =~ s/ von der / von~der~/g;
  # Known "bugs" noted later:
  # (a) Forgot O!
  # (b) should have had the multi-part (e.g. 'van der ') occur *before* the 'van' on its own
  # But NOTE: we keep these bugs in place here because all the cluster data was processed with these bugs in place
  foreach my $part (split(/[ -]/, $name)) {
	next if ($part eq "");
	next if ($part eq "mysteriously");
	next if ($part eq "unnamed");
	next if ($part =~ /^the$/i);
	next if ($part =~ /^news$/i);
	next if ($part =~ /~$/);	# But can't end or start on a '~'
	next if ($part =~ /^~/);
	next if ($part =~ /[0-9]/); # Can't contain a number
	next if ($part =~ /^[a-z]$/); # Nothing that is a single letter [but could have single characters]
	next if ($part eq "Jr" || $part eq "Sr" ||
			 $part eq "jr" || $part eq "sr" ||
			 $part eq "JR" || $part eq "SR" ||
			 $part eq "junior" || $part eq "senior" ||
			 $part eq "Junior" || $part eq "Senior" ||
			 $part eq "JUNIOR" || $part eq "SENIOR"
			);
	push(@nameParts, $part);
  }

  my $size = scalar(@nameParts);

  print join(" ", @nameParts), "\n";
}

