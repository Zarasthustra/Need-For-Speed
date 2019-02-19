################################################################################
#
#    A general struct (Table) and two functions vector1D and vector.
#        - vector1D and vector are taken from Molly
#     https://github.com/jgreener64/Molly.jl/blob/master/src/md.jl
################################################################################
using Distributed
using Base.Threads
using BenchmarkTools
using StaticArrays
using StructArrays
using Setfield
using LinearAlgebra: norm, normalize, dot, ×

struct Table  # given dummy values below
    ϵij::Array{Float64,2}
    σij::Array{Float64,2}
end
mutable struct XYZ <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end
mutable struct ArrayOfStructs
    r::XYZ
    v::XYZ
    f::XYZ
    type::Int64
end

struct ArrayOfSArrayStructs{T}
    r::SVector{3,T}
    v::SVector{3,T}
    f::SVector{3,T}
    type::Int64
end

mutable struct StructOfArrays
    r::Vector{XYZ}
    v::Vector{XYZ}
    f::Vector{XYZ}
    type::Vector{Int64}
end

mutable struct StructOfSArrays
    r::Vector{SVector{3,Float64}}
    v::Vector{SVector{3,Float64}}
    f::Vector{SVector{3,Float64}}
    type::Vector{Int64}
end

"Vector between two coordinate values, accounting for mirror image seperation"
@inline function vector1D(c1::Float64, c2::Float64, box_size::Float64)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + box_size) ? (c2 - c1) : (c2 - c1 - box_size)
    else
        return (c1 - c2) < (c2 - c1 + box_size) ? (c2 - c1) : (c2 - c1 + box_size)
    end
end

vector(coords_one::SVector{3,Float64}, coords_two::SVector{3,Float64}, box_size::Float64) =
         [vector1D(coords_one[1], coords_two[1], box_size),
        vector1D(coords_one[2], coords_two[2], box_size),
        vector1D(coords_one[3], coords_two[3], box_size)]

@inline function SingleLJ2!(f1::XYZ, f2::XYZ,
                    coord1::XYZ,coord2::XYZ, ϵ::Float64, σ::Float64,
                    box_size::SVector{3,Float64})

    dx = vector1D(coord1.x, coord2.x, box_size[1])
    dy = vector1D(coord1.y, coord2.y, box_size[2])
    dz = vector1D(coord1.z, coord2.z, box_size[3])

    rij_sq = dx*dx + dy*dy + dz*dz

    if  rij_sq > 1.0
        return
    else
        sr2 = σ^2 / rij_sq
        sr6 = sr2 ^ 3
        sr12 = sr6 ^ 2
        f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

        x = -f * dx
        y = -f * dy
        z = -f * dz

        f1.x += x
        f1.y += y
        f1.z += z
        f2.x -= x
        f2.y -= y
        f2.z -= z
    end
    nothing
end

function TotalEnergyStructOfArrays(array::StructOfArrays,vdwTable::Table,
                                   n::Int64,box_size::SVector{3,Float64})

    @inbounds for i = 1:(n-1)
        ti = array.type[i]
        for j = (i+1):n
            tj = array.type[j]
            SingleLJ2!(array.f[i], array.f[j],
                       array.r[i], array.r[j],
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size)
        end
    end
    return
end

@inline function SingleLJ!(f1::XYZ,f2::XYZ,
        coord1::XYZ,coord2::XYZ, ϵ::Float64, σ::Float64,
        box_size::SVector{3,Float64})


    # method 4 2 allocations: 224 bytes
    dx = vector1D(coord1.x, coord2.x, box_size[1])
    dy = vector1D(coord1.y, coord2.y, box_size[2])
    dz = vector1D(coord1.z, coord2.z, box_size[3])

    rij_sq = dx*dx + dy*dy + dz*dz  # use only with method 4

    if  rij_sq > 1.0
        return
    else

        sr2 = σ^2 / rij_sq
        sr6 = sr2 ^ 3
        sr12 = sr6 ^ 2
        f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

        x = -f * dx
        y = -f * dy
        z = -f * dz

        f1.x += x
        f1.y += y
        f1.z += z
        f2.x -= x
        f2.y -= y
        f2.z -= z
    end
    nothing
end
function TotalEnergyArrayOfStructs(array::Vector{ArrayOfStructs},vdwTable::Table,
                        n::Int64,box_size::SVector{3,Float64})

    @inbounds for i = 1:(n-1)
        ti = array[i].type
        for j = (i+1):n
            tj = array[j].type
            SingleLJ!(array[i].f, array[j].f,
                       array[i].r, array[j].r,
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size)

        end
    end

    return
end

################################################################################
#              Marray tests
################################################################################
@inline function SingleLJ3!(f1::MVector{3,Float64},f2::MVector{3,Float64},
        diff::SVector{3,Float64},
        coord1::MVector{3,Float64},coord2::MVector{3,Float64}, ϵ::Float64,
        σ::Float64, box_size::SVector{3,Float64})

        @inbounds for i=1:3
         diff = @set diff[i] = vector1D(coord1[i], coord2[i], box_size[i])
        end

        #rij_sq = dx*dx  + dy*dy + dz*dz
        rij_sq = diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3]

        if  rij_sq > 1.0
            return
        else

            sr2 = σ^2 / rij_sq
            sr6 = sr2 ^ 3
            sr12 = sr6 ^ 2
            f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

            @inbounds for i = 1:3
                f1[i] += -f * diff[i]
            end
            #f1[1] += -f * dx
            #f1[2] += -f * dx
            #f1[3] += -f * dx
            @inbounds for i in eachindex(f1)
                f2[i] -= -f * diff[i]
            end
        end
    nothing
end
function TotalEnergySarray(r::Array{MArray{Tuple{3},Float64,1,3},1},f::Array{MArray{Tuple{3},Float64,1,3},1},
                        type::Array{MArray{Tuple{1},Int64,1,1},1}, vdwTable::Table,
                        n::Int64,box_size::SVector{3,Float64})
    diff = SVector{3}(0.0,0.0,0.0)
    @inbounds for i = 1:(n-1)
        ti = type[i][1]
        for j = (i+1):n
            tj = type[j][1]
            SingleLJ3!(f[i], f[j], diff,
                      r[i], r[j],
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size)

        end
    end

    return
end
################################################################################
#              MMatrix tests
################################################################################
@inline function SingleLJ4!(f1::SubArray,f2::SubArray,
        diff::SVector{3,Float64},
        coord1::SubArray{Float64,1},coord2::SubArray{Float64,1},
         ϵ::Float64, σ::Float64, box_size::SVector{3,Float64}) where S

    #dx = vector1D(coord1[1], coord2[1], box_size[1])
    #dy = vector1D(coord1[2], coord2[2], box_size[2])
    #dz = vector1D(coord1[3], coord2[3], box_size[3])

    #rij_sq = dx*dx + dy*dy + dz*dz  # use only with method 4

    @inbounds for i=1:3
        diff = @set diff[i] = vector1D(coord1[i], coord2[i], box_size[i])
    end

    #rij_sq = dx*dx  + dy*dy + dz*dz
    rij_sq = diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3]

    if  rij_sq > 1.0
        return
    else

        sr2 = σ^2 / rij_sq
        sr6 = sr2 ^ 3
        sr12 = sr6 ^ 2
        f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

        @inbounds for i = 1:3
            f1[i] += -f * diff[i]
        end
        #f1[1] += -f * dx
        #f1[2] += -f * dx
        #f1[3] += -f * dx
        @inbounds for i in eachindex(f1)
            f2[i] += f * diff[i]
        end
    end
    nothing
end
function TotalEnergyMMatrix(r::MArray{Tuple{T,3},Float64,2,L},f::MArray{Tuple{T,3},Float64,2,L},
                        type::Array{MArray{Tuple{1},Int64,1,1},1}, vdwTable::Table,
                        n::Int64,box_size::SVector{3,Float64}) where T where L
    diff = SVector{3}(0.0,0.0,0.0)
    @inbounds for i = 1:(n-1)
        ti = type[i][1]
        for j = (i+1):n
            tj = type[j][1]
            SingleLJ4!(f[i,1:3],f[j,1:3],diff, #f1, f2,
                      @view(r[i,1:3]), @view(r[j,1:3]),
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size)
        end
    end

    return
end
################################################################################
#              @Sarray tests
################################################################################
@inline function SingleLJ5!(f1::Vector{Float64},f2::Vector{Float64},
        diff::SVector{3,Float64},
        coord1::SubArray{Float64,1},coord2::SubArray{Float64,1},
         ϵ::Float64, σ::Float64, box_size::SVector{3,Float64})

        @inbounds for i=1:3
         diff = @set diff[i] = vector1D(coord1[i], coord2[i], box_size[i])
        end

        #rij_sq = dx*dx  + dy*dy + dz*dz
        rij_sq = diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3]
#=
    dx = vector1D(coord1[1], coord2[1], box_size[1])
    dy = vector1D(coord1[2], coord2[2], box_size[2])
    dz = vector1D(coord1[3], coord2[3], box_size[3])

    rij_sq = dx*dx + dy*dy + dz*dz  # use only with method 4
=#
    if  rij_sq > 1.0
        return
    else

        sr2 = σ^2 / rij_sq
        sr6 = sr2 ^ 3
        sr12 = sr6 ^ 2
        f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

        @inbounds for i = 1:3
            f1[i] += -f * diff[i]
        end
        #f1[1] += -f * dx
        #f1[2] += -f * dx
        #f1[3] += -f * dx
        @inbounds for i in eachindex(f1)
            f2[i] -= -f * diff[i]
        end
    end
    nothing
end
function TotalEnergyAtSArray(r::SArray{Tuple{S,R},T,N,L} ,f::SArray{Tuple{S,R},T,N,L} ,
                        type::SArray{Tuple{S}}, vdwTable::Table,
                        n::Int64,box_size::SVector{3,Float64}) where S where R where L where N where T
    #f1 = Vector{Float64}(undef,3)
    #f2 = Vector{Float64}(undef,3)
    diff = SVector{3}(0.0,0.0,0.0)
    @inbounds for i = 1:(n-1)
        ti = type[i][1]
        for j = (i+1):n
            tj = type[j][1]
            SingleLJ5!(f[i,1:3],f[j,1:3],diff, #f1, f2,
                      @view(r[i,1:3]), @view(r[j,1:3]),
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size)

        end
    end

    return
end

################################################################################
#                    Struct of SArrays
################################################################################

@inline function SingleLJ6!(diff::SVector{3,Float64},#f2::SVector{3,Float64},
                    f1::SVector{3,Float64}, f2::SVector{3,Float64},
                    coord1::SVector{3,Float64},coord2::SVector{3,Float64},
                    ϵ::Float64, σ::Float64, box_size::SVector{3,Float64},
                    #diff::SVector{3,Float64}
                    )

    @inbounds for i=1:3
        diff = @set diff[i] = vector1D(coord1[i], coord2[i], box_size[i])
    end

    rij_sq = diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3]

    if  rij_sq > 1.0
        return zero(diff), zero(diff)
    else

        sr2 = σ ^ 2 / rij_sq
        sr6 = sr2 ^ 3
        sr12 = sr6 ^ 2
        f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

    end

    @inbounds for i = 1:3
        f1 = @set f1[i] += -f * diff[i]
        f2 = @set f2[i] +=  f * diff[i]
    end
    return f1,f2
end

function TotalEnergyStructOfSArrays(array::StructOfSArrays,vdwTable::Table,
                                   n::Int64,box_size::SVector{3,Float64})

    diff = SVector{3}(0.0,0.0,0.0)

    @inbounds for i = 1:(n-1)
        ti = array.type[i]
        for j = (i+1):n
            tj = array.type[j]
            array.f[i],array.f[j] = SingleLJ6!(diff,
                       array.f[i],array.f[j],
                       array.r[i], array.r[j],
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size) #,)

        end
    end

    return
end

################################################################################
#                    Git Struct Array
################################################################################

@inline function SingleLJ7!(diff::SVector{3,Float64},
                    f1::SArray{Tuple{3},Float64,1,3}, f2::SArray{Tuple{3},Float64,1,3},
                    coord1::SVector{3,Float64},
                    coord2::SVector{3,Float64},
                    ϵ::Float64, σ::Float64, box_size::SVector{3,Float64},
                    )

    @inbounds for i=1:3
        diff = setindex(diff,vector1D(coord1[i], coord2[i], box_size[i]),i)
    end

    rij_sq = diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3]

    if  rij_sq > 1.0

        return zero(diff),zero(diff)
    else
        sr2 = σ^2 / rij_sq
        sr6 = sr2 ^ 3
        sr12 = sr6 ^ 2
        f = ((24 * ϵ) / rij_sq) * (2 * sr12 - sr6)

    end
  #=
    @inbounds for i = 1:3
        f1 = @set f1[i] += -f * diff[i]
        f2 = @set f2[i] +=  f * diff[i]
    end
     =#

    diff = f * diff

    for i = 1:3
        f1 = setindex(f1,f1[i] - diff[i],i)    #+= -f * diff
        f2 = setindex(f2,f2[i] + diff[i],i)    #+= f * diff
    end

    return f1,f2
    #nothing
end

function TotalEnergyGitStructArray(
    array::StructArray{ArrayOfSArrayStructs{Float64},1,NamedTuple{(:r, :v, :f, :type),Tuple{Array{SArray{Tuple{3},Float64,1,3},1},Array{SArray{Tuple{3},Float64,1,3},1},Array{SArray{Tuple{3},Float64,1,3},1},Array{Int64,1}}}},
                                   vdwTable::Table,
                                   n::Int64,box_size::SVector{3,Float64})
    # Uses the StructArray package, need to figure out a neater way to pass "array" in

    diff = SVector{3}(0.0,0.0,0.0) # preallocate

    @inbounds for i = 1:(n-1)
        ti = array.type[i]
        for j = (i+1):n
            tj = array.type[j]
            #f1,f2 = SingleLJ7!(diff,
            array.f[i], array.f[j] = SingleLJ7!(diff,
            #SingleLJ7!(diff,
                       array.f[i], array.f[j],
                       array.r[i], array.r[j],
                       vdwTable.ϵij[ti,tj],
                       vdwTable.σij[ti,tj],
                       box_size) #,)
            #array = @set array.f[i] = f1
            #array = @set array.f[j] = f2

        end
    end
    return
end
################################################################################
#                  Some general parameters used by all trials
################################################################################

box_size = SVector{3}(2.0,2.0,2.0)
x = 3011 # dummy number of atoms
m = 4    # dummy number of atom types (for table of parameters...)
println("This program is using ", Threads.nthreads(), " threads")
vdwTable = Table([rand() for i=1:m,j=1:m],
            [rand() for i=1:m,j=1:m]) #,
################################################################################
#                   Start of trials
################################################################################
" make static but mutable arrays "
r = [MVector{3}(rand(), rand(), rand()) .* box_size[1] for i = 1:x]
f = [MVector{3}(0.0, 0.0, 0.0) for i = 1:x]
type = [MVector{1}(rand(1:m)) for i = 1:x] #MVector{x}(rand(1:m),x) #for i=1:x

" make static but mutable matrices "
mr = zeros(MMatrix{x,3})
mf = zeros(MMatrix{x,3})

for i=1:x, j=1:3
    mr[i,j] = rand()
end

" Static Immutable Matrix's... crashes my computer :S "
#sA_r  = SMatrix{x,3}([rand() for i = 1:(x*3)]) #@SArray rand(x,3)
#sA_f  = SMatrix{x,3}([rand() for i = 1:(x*3)])
#sA_t  = [SVector{1}(rand(1:m)) for i = 1:x]
#sA_f_old = deepcopy(sA_f)
#sA_f = @SArray [rand() for i = 1:x, j=1:3]

"Array Of Structs approach, uses XYZ for vectors"

arrayOfStructs = Vector{ArrayOfStructs}(undef,x)
"Array of Structs, uses static vectors for vectors"

arrayOfSOA = Vector{ArrayOfSArrayStructs}(undef,x)


arrayOfStructs = [ArrayOfStructs( XYZ(box_size[1] * rand(), box_size[2] * rand(), box_size[3] * rand() ),
                                    XYZ(0.0, 0.0, 0.0),
                                    XYZ(0.0, 0.0, 0.0),
                                    ceil(rand()*m) ) for i=1:x]

arrayOfSOA = [ArrayOfSArrayStructs( box_size[1] * rand(SVector{3}), rand(SVector{3}), zeros(SVector{3}), rand(1:m) ) for i=1:x]

#"Convert to StructArray using package StructArray"
SoA = StructArray(arrayOfSOA)

println("TYPE: ", typeof(SoA) )
#deepcopyAoS = deepcopy(arrayOfStructs)
"Struct of Arrays using XYZ vectors"

@time structOfArrays = StructOfArrays([XYZ(box_size[1] * rand(), box_size[2] * rand(), box_size[3] * rand()) for i=1:x],
                [XYZ(0.0, 0.0, 0.0) for i=1:x ],
                [XYZ(0.0, 0.0, 0.0) for i=1:x ],
                [ceil(rand())*m for i=1:x],)

"Struct of Arrays using static vectors"

@time structOfSArrays = StructOfSArrays(
                [SVector{3}(box_size[1] * rand(), box_size[1] * rand(), box_size[1] * rand()) for i=1:x],
                [SVector{3}(0.0, 0.0, 0.0) for i=1:x ],
                [SVector{3}(0.0, 0.0, 0.0) for i=1:x ],
                SVector{x}([ceil(rand())*m for i=1:x]),
                )


################################################################################
#
#                           Testing Github StructArray
#
################################################################################

println("--------------------Start of Testing StructArray-------------------------")
print("StructArray: ")
#TotalEnergyGitStructArray(SoA.r,SoA.f,SoA.type,vdwTable,x,box_size)
#@btime TotalEnergyGitStructArray($SoA.r,$SoA.f,$SoA.type,$vdwTable,$x,$box_size)
#@code_warntype TotalEnergyGitStructArray(SoA,vdwTable,x,box_size)
#@code_native TotalEnergyGitStructArray(SoA,vdwTable,x,box_size)
TotalEnergyGitStructArray(SoA,vdwTable,x,box_size)
@btime TotalEnergyGitStructArray($SoA,$vdwTable,$x,$box_size)
#@profiler TotalEnergyGitStructArray(SoA,vdwTable,x,box_size)
println("--------------------Start of Testing StructArray-------------------------")

################################################################################
#
#                           Testing Struct of SArrays
#
################################################################################

println("------------------------Start of Testing SoSA------------------------------")
print("Struct of Static Arrays: ")
@btime TotalEnergyStructOfSArrays($structOfSArrays,$vdwTable,$x,$box_size)
#@profiler TotalEnergyStructOfSArrays(structOfSArrays,vdwTable,x,box_size)
println("--------------------------End of Testing SoSA------------------------------")

################################################################################
#
#         Testing Array of Structs
#
################################################################################

TotalEnergyArrayOfStructs(arrayOfStructs,vdwTable,x,box_size)  # precompile
println("------------------------Start of Testing AoS------------------------------")
print("array of structs (XYZ): ")
@btime TotalEnergyArrayOfStructs($arrayOfStructs,$vdwTable,$x,$box_size)
println("--------------------------End of Testing AoS------------------------------")

################################################################################
#
#                           Testing Struct of Arrays
#
################################################################################

println("------------------------Start of Testing SoA------------------------------")
print("Struct of Arrays (XYZ): ")
@btime TotalEnergyStructOfArrays($structOfArrays,$vdwTable,$x,$box_size)
println("--------------------------End of Testing SoA------------------------------")

################################################################################
#
#                           Testing Static Arrays
#
################################################################################
println("----------------------Start of Testing SArrays----------------------------")
print("Static MArrays  : ")
#TotalEnergySarray(r,f,type,vdwTable,x,box_size)
@btime TotalEnergySarray($r,$f,$type,$vdwTable,$x,$box_size)
println("------------------------End of Testing SArrays----------------------------")


################################################################################
#
#                           Testing MMatrix
#
################################################################################
println("----------------------Start of Testing MMAtrix----------------------------")
print("MMatrix      : ")
#TotalEnergyMMatrix(mr,mf,type,vdwTable,x,box_size)
@btime TotalEnergyMMatrix($mr,$mf,$type, $vdwTable, $x,$box_size)
println("------------------------End of Testing MMAtrix----------------------------")

################################################################################
#
#                           Testing @Sarray
#
################################################################################
println("----------------------Start of Testing @Sarray----------------------------")
print("@Sarray      : ")
#TotalEnergyMMatrix(mr,mf,type,vdwTable,x,box_size)
#@btime TotalEnergyAtSArray($sA_r,$sA_f,$sA_t, $vdwTable, $x,$box_size)
println("------------------------End of Testing @Sarray----------------------------")

#end

#main()
